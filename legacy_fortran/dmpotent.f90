module dmpotent
    use numeric_kinds
    use initial_parameters
    implicit none
    private

    ! Generic module for black hole mass and dark halo potential
    ! calls triaxpotent

    ! RvdB 17 Mar 2010

    ! Bugs

    ! calculate potential phi at (x,y,z)
    public:: dm_potent

    ! calculate accel ax,ay,ax at (x,y,z)
    public:: dm_accel

    ! setup the constants for the potential
    public:: dm_setup

    ! setup the constants for the potential
    public:: dm_stop

    ! sBH-only entry points, for the standalone agreement probe. These
    ! duplicate the sBH blocks of dm_setup/dm_potent/dm_accel but call the
    ! same sbh_menc/sbh_outer_tail helpers, so they cannot drift.
    public:: dm_setup_sbh_only, dm_potent_sbh_only, dm_accel_sbh_only

    real(kind=dp), private :: rhoc, rc, dm_logslp

    ! sBH subcluster (Zhao alpha-beta-gamma), kept in its own slot so the
    ! existing halo cases 1,2,3,5 are untouched.
    logical, private :: sbh_present = .false.
    real(kind=dp), private :: sbh_rho0, sbh_a, sbh_al, sbh_be, sbh_ga

    ! Tolerance and hard cap shared by the two series below. Both converge
    ! geometrically, so the tolerance is what stops them and the cap only
    ! ever fires outside the documented domain -- where it aborts rather
    ! than returning a silently truncated sum. Mirrors _SERIES_TOL and
    ! _SERIES_MAXIT in dynamite/physical_system.py:StellarBlackHoles.
    real(kind=dp), parameter, private :: sbh_series_tol = 1.0e-16_dp
    integer(kind=i4b), parameter, private :: sbh_series_maxit = 100000

    ! Floor applied to y = (r/a)**alpha when it underflows to zero; see
    ! sbh_yvar for why this is clamped rather than treated as an error.
    real(kind=dp), parameter, private :: sbh_y_floor = 1.0e-300_dp
    logical, private :: sbh_warned_underflow = .false.

contains

    ! -----------------------------------------------------------------
    ! sBH helpers. Everything here is a port of the corresponding routine
    ! in dynamite/physical_system.py:StellarBlackHoles; keep the two in
    ! step. The natural variable throughout is y = (r/a)**alpha -- never
    ! 1-x, which rounds to exactly 1.0 for small r and destroys the answer.
    !
    ! Note these deliberately do NOT use zh_betai: its continued fraction
    ! (sub/specfunc_beta.f90, zh_betacf) carries EPS=3e-7, i.e. single
    ! precision, while the Python reference uses scipy's double-precision
    ! betainc. The series below is double throughout.
    ! -----------------------------------------------------------------

    ! (exp(z)-1)/z, accurate as z -> 0 where it tends to 1. Fortran has no
    ! expm1 intrinsic and the Python reference has np.expm1, so we need one.
    !
    ! A plain (exp(z)-1)/z loses ~eps/|z| to cancellation: 5e-12 at
    ! z = 4e-5, measured as a 5.76e-13 error in the potential at
    ! (alpha,beta,gamma) = (3,4,1.99999). So small |z| needs its own branch.
    !
    ! That branch is the Maclaurin series sum_n z**n/(n+1)!, taken to
    ! convergence, NOT Kahan's (exp(z)-1)/z = (u-1)/log(u). Kahan is more
    ! elegant and is correct under IEEE arithmetic, but it relies on the
    ! rounding errors of u-1 and log(u) cancelling, and -ffast-math -- which
    ! this makefile passes -- assumes log(exp(z)) == z and folds the identity
    ! away. Measured: Kahan gives 7.35e-16 at -O2 and 5.76e-13 under the
    ! production flags, i.e. no better than the naive form it replaced. The
    ! series has no cancellation to protect and so cannot be optimised away.
    ! For |z| >= 1/2 the direct form is fine (|exp(z)-1| >= 0.39).
    function sbh_expm1_over_z(z) result(val)
        real(kind=dp), intent(in) :: z
        real(kind=dp) :: val, term
        integer(kind=i4b) :: n

        if (abs(z) .ge. 0.5_dp) then
            val = (exp(z) - 1.0_dp)/z
            return
        end if
        term = 1.0_dp
        val = 1.0_dp
        do n = 1, 40
            term = term*z/real(n + 1, dp)
            val = val + term
            if (abs(term) .le. 1.0e-18_dp*abs(val)) exit
        end do
    end function sbh_expm1_over_z

    ! Complete beta B(p,q), p,q > 0. log_gamma is an F2008 intrinsic and
    ! is permitted by the -std=legacy build.
    function sbh_cbeta(p, q) result(val)
        real(kind=dp), intent(in) :: p, q
        real(kind=dp) :: val

        val = exp(log_gamma(p) + log_gamma(q) - log_gamma(p + q))
    end function sbh_cbeta

    ! B(x;p,q) = int_0^x u**(p-1) (1-u)**(q-1) du by the series about
    ! x = 0, valid for 0 <= x < 1, p > 0, q of either sign:
    !     B(x;p,q) = x**p * Sum_k e_k x**k/(p+k),  e_k = e_{k-1}(k-q)/k
    ! Stopping is a tail bound, not a term count, and only after k > |q|
    ! where the coefficients have stopped growing.
    function sbh_beta_series(x, p, q) result(val)
        real(kind=dp), intent(in) :: x, p, q
        real(kind=dp) :: val, total, coef, term
        integer(kind=i4b) :: k

        if (x .le. 0.0_dp) then
            val = 0.0_dp
            return
        end if
        total = 1.0_dp/p
        coef = 1.0_dp
        do k = 1, sbh_series_maxit
            coef = coef*(real(k, dp) - q)/real(k, dp)
            term = coef*x**k/(p + real(k, dp))
            total = total + term
            if (real(k, dp) .gt. abs(q) .and. &
                abs(term)*x .le. (1.0_dp - x)*sbh_series_tol*abs(total)) exit
        end do
        if (k .gt. sbh_series_maxit) then
            print *, 'sbh_beta_series: no convergence, x,p,q =', x, p, q
            stop 'sbh_beta_series: series did not converge'
        end if
        val = x**p*total
    end function sbh_beta_series

    ! Continued fraction for the incomplete beta, Lentz's method. This is
    ! Numerical Recipes' betacf, but in genuine double precision: the
    ! zh_betacf in sub/specfunc_beta.f90 carries EPS = 3e-7 and MAXIT = 500,
    ! which is single precision and cannot match the scipy reference.
    ! Requires a, b > 0 and 0 <= x <= 1.
    function sbh_betacf(a, b, x) result(h)
        real(kind=dp), intent(in) :: a, b, x
        real(kind=dp) :: h, aa, c, d, del, qab, qam, qap
        real(kind=dp), parameter :: eps = 3.0e-16_dp, fpmin = 1.0e-300_dp
        integer(kind=i4b), parameter :: maxit = 1000
        integer(kind=i4b) :: m, m2

        qab = a + b
        qap = a + 1.0_dp
        qam = a - 1.0_dp
        c = 1.0_dp
        d = 1.0_dp - qab*x/qap
        if (abs(d) .lt. fpmin) d = fpmin
        d = 1.0_dp/d
        h = d
        do m = 1, maxit
            m2 = 2*m
            aa = real(m, dp)*(b - real(m, dp))*x &
                 /((qam + real(m2, dp))*(a + real(m2, dp)))
            d = 1.0_dp + aa*d
            if (abs(d) .lt. fpmin) d = fpmin
            c = 1.0_dp + aa/c
            if (abs(c) .lt. fpmin) c = fpmin
            d = 1.0_dp/d
            h = h*d*c
            aa = -(a + real(m, dp))*(qab + real(m, dp))*x &
                 /((a + real(m2, dp))*(qap + real(m2, dp)))
            d = 1.0_dp + aa*d
            if (abs(d) .lt. fpmin) d = fpmin
            c = 1.0_dp + aa/c
            if (abs(c) .lt. fpmin) c = fpmin
            d = 1.0_dp/d
            del = d*c
            h = h*del
            if (abs(del - 1.0_dp) .le. eps) exit
        end do
        if (m .gt. maxit) then
            print *, 'sbh_betacf: no convergence, a,b,x =', a, b, x
            stop 'sbh_betacf: continued fraction did not converge'
        end if
    end function sbh_betacf

    ! B(x;p,q) for p,q > 0, with the complement xc = 1-x supplied by the
    ! caller so it is never formed by subtraction. Numerical Recipes'
    ! branch selection: the continued fraction is evaluated on whichever
    ! side of x = (p+1)/(p+q+2) converges, which is also the side where
    ! the answer is not a small difference of large numbers.
    !
    ! The series route is deliberately NOT used here. Its coefficients
    ! alternate for q > 0 and Sum|e_k| x**k grows like (1-x)**-q, so it
    ! silently loses q*log10(1/xc) digits -- measured at 3e-10 for
    ! (alpha,beta,gamma) = (0.2,12,2.9) -- and reflecting to dodge that
    ! just moves the loss into B(p,q) minus a nearly equal number
    ! (1e17 relative error at (0.15,20,-5)). The continued fraction has
    ! neither problem. The series survives only where the continued
    ! fraction cannot go: q <= 0, in sbh_outer_tail_integral.
    function sbh_binc(x, xc, p, q) result(val)
        real(kind=dp), intent(in) :: x, xc, p, q
        real(kind=dp) :: val, bt

        if (x .le. 0.0_dp) then
            val = 0.0_dp
            return
        end if
        if (xc .le. 0.0_dp) then
            val = sbh_cbeta(p, q)
            return
        end if
        bt = x**p*xc**q
        if (x .lt. (p + 1.0_dp)/(p + q + 2.0_dp)) then
            val = bt*sbh_betacf(p, q, x)/p
        else
            val = sbh_cbeta(p, q) - bt*sbh_betacf(q, p, xc)/q
        end if
    end function sbh_binc

    ! y = (r/a)**alpha, floored. See the design note in the task report:
    ! underflow needs r/a below ~1e-79 for the fitted exponents, which no
    ! orbit reaches, and aborting an orbit library mid-integration is a
    ! far worse failure mode than a clamped (still enormous) potential.
    function sbh_yvar(r) result(y)
        real(kind=dp), intent(in) :: r
        real(kind=dp) :: y

        y = (r/sbh_a)**sbh_al
        if (.not. (y .gt. sbh_y_floor)) then
            if (.not. sbh_warned_underflow) then
                print *, 'WARNING: sBH (r/a)**alpha underflowed at r =', r, &
                    ' - clamped to', sbh_y_floor
                sbh_warned_underflow = .true.
            end if
            y = sbh_y_floor
        end if
    end function sbh_yvar

    ! int_y^inf s**(q-1) (1+s)**-(p+q) ds, for y > 0 and p > 0. This is
    ! the potential's outer term in the variable y. Three branches,
    ! mirroring _outer_tail_integral in the Python.
    function sbh_outer_tail_integral(y, p, q) result(val)
        real(kind=dp), intent(in) :: y, p, q
        real(kind=dp) :: val, w, u, c, y_c, log_ratio, total, coef, term
        real(kind=dp) :: e, zz, yc_e
        integer(kind=i4b) :: k

        u = 1.0_dp/(1.0_dp + y)
        if (q .gt. 1.0_dp) then
            ! Comfortably finite at y = 0, and both of the Python's
            ! sub-cases are just B(x;p,q) with x = 1/(1+y). sbh_binc takes
            ! the complement exactly as y/(1+y) rather than through
            ! 1 - 1/(1+y), and its own branch test already picks the side
            ! that avoids subtracting near-equal numbers, so the two are
            ! collapsed into one call here.
            w = y/(1.0_dp + y)
            val = sbh_binc(u, w, p, q)
            return
        end if
        c = p + q
        y_c = min(0.5_dp, 1.0_dp/c)
        if (y .ge. y_c) then
            val = sbh_beta_series(u, p, q)
            return
        end if
        ! Split at s = y_c: the upper piece is a constant, the lower one
        ! expands (1+s)**-(p+q) binomially. The k = 0 term carries the
        ! divergent y**q/q and is built from y directly.
        log_ratio = log(y/y_c)
        total = sbh_beta_series(1.0_dp/(1.0_dp + y_c), p, q)
        coef = 1.0_dp
        do k = 0, sbh_series_maxit
            if (k .gt. 0) coef = -coef*(c + real(k, dp) - 1.0_dp)/real(k, dp)
            e = q + real(k, dp)
            zz = e*log_ratio
            yc_e = y_c**e
            if (abs(zz) .lt. 1.0_dp) then
                ! regular form: no cancellation, and safe at e == 0
                term = -coef*yc_e*log_ratio*sbh_expm1_over_z(zz)
            else
                term = coef*(yc_e - y**e)/e
            end if
            total = total + term
            if (real(k, dp) .gt. c .and. &
                abs(term)*y_c .le. (1.0_dp - y_c)*sbh_series_tol*abs(total)) exit
        end do
        if (k .gt. sbh_series_maxit) then
            print *, 'sbh_outer_tail_integral: no convergence, y,p,q =', y, p, q
            stop 'sbh_outer_tail_integral: series did not converge'
        end if
        val = total
    end function sbh_outer_tail_integral

    ! 4 pi int_r^inf r' rho(r') dr', the potential's outer term. Accepts an
    ! already-computed y = (r/a)**alpha via y_in, so a caller that also
    ! needs sbh_menc(r) at the same r (dm_potent's sBH block) evaluates the
    ! (r/a)**alpha power once instead of once per function.
    function sbh_outer_tail(r, y_in) result(tail)
        real(kind=dp), intent(in) :: r
        real(kind=dp), intent(in), optional :: y_in
        real(kind=dp) :: tail, y, p_out, q_out

        y = sbh_yvar(r)
        if (present(y_in)) y = y_in
        p_out = (sbh_be - 2.0_dp)/sbh_al
        q_out = (2.0_dp - sbh_ga)/sbh_al
        tail = 4.0_dp*pi_d*sbh_a*sbh_a*sbh_rho0/sbh_al &
               *sbh_outer_tail_integral(y, p_out, q_out)
    end function sbh_outer_tail

    ! M(<r) for the sBH profile, in Msun. Both beta parameters are
    ! strictly positive given gamma < 3 and beta > 3. See sbh_outer_tail
    ! for the optional y_in argument.
    function sbh_menc(r, y_in) result(menc)
        real(kind=dp), intent(in) :: r
        real(kind=dp), intent(in), optional :: y_in
        real(kind=dp) :: menc, y, w, u

        y = sbh_yvar(r)
        if (present(y_in)) y = y_in
        w = y/(1.0_dp + y)
        u = 1.0_dp/(1.0_dp + y)
        menc = 4.0_dp*pi_d*sbh_a**3*sbh_rho0/sbh_al &
               *sbh_binc(w, u, (3.0_dp - sbh_ga)/sbh_al, &
                         (sbh_be - 3.0_dp)/sbh_al)
    end function sbh_menc

    ! Total sBH mass, in Msun. Taken from the complete beta rather than
    ! from sbh_menc at some large radius, whose y = (r/a)**alpha overflows
    ! to infinity for large alpha.
    function sbh_mtot() result(mtot)
        real(kind=dp) :: mtot

        mtot = 4.0_dp*pi_d*sbh_a**3*sbh_rho0/sbh_al &
               *sbh_cbeta((3.0_dp - sbh_ga)/sbh_al, (sbh_be - 3.0_dp)/sbh_al)
    end function sbh_mtot

    ! Common validation/assignment for the sBH slot.
    subroutine sbh_assign()
        ! sbhparam is unallocated whenever the block was absent from the
        ! input file, and a caller that sets sbh_profile_type by hand (the
        ! Task 7 probe) can reach here without allocating it. Diagnose that
        ! rather than reading out of bounds.
        if (.not. allocated(sbhparam)) stop 'sBH parameters not allocated'
        if (size(sbhparam) .lt. 5) stop 'sBH parameter array too small'
        if (n_sbhparam .ne. 5) stop 'wrong number of sBH parameters'
        sbh_rho0 = sbhparam(1)   ! Msun/km^3
        sbh_a = sbhparam(2)      ! km
        sbh_al = sbhparam(3)
        sbh_be = sbhparam(4)
        sbh_ga = sbhparam(5)
        if (sbh_rho0 .le. 0.0_dp) stop 'sBH rho0 must be > 0'
        if (sbh_a .le. 0.0_dp) stop 'sBH scale radius must be > 0'
        if (sbh_al .le. 0.0_dp) stop 'sBH alpha must be > 0'
        if (sbh_be .le. 3.0_dp) stop 'sBH beta must be > 3 (finite mass)'
        if (sbh_ga .ge. 3.0_dp) stop 'sBH gamma must be < 3'
        if (abs(sbh_ga - 2.0_dp) .lt. 1.0e-6_dp) &
            stop 'sBH gamma must not equal 2'
    end subroutine sbh_assign

    ! sBH-only entry points for the agreement probe (Task 7). They share
    ! sbh_menc / sbh_outer_tail with the real code paths above.
    subroutine dm_setup_sbh_only()
        sbh_present = (sbh_profile_type .eq. 6)
        if (.not. sbh_present) stop 'dm_setup_sbh_only: no sBH block'
        call sbh_assign()
    end subroutine dm_setup_sbh_only

    subroutine dm_potent_sbh_only(x, y, z, pot)
        real(kind=dp), intent(in) :: x, y, z
        real(kind=dp), intent(out) :: pot
        real(kind=dp) :: d, yv

        d = sqrt(x*x + y*y + z*z)
        yv = sbh_yvar(d)
        pot = grav_const_km*(sbh_menc(d, yv)/d + sbh_outer_tail(d, yv))
    end subroutine dm_potent_sbh_only

    subroutine dm_accel_sbh_only(x, y, z, vx, vy, vz)
        real(kind=dp), intent(in) :: x, y, z
        real(kind=dp), intent(out) :: vx, vy, vz
        real(kind=dp) :: d, acceleration_r

        d = sqrt(x*x + y*y + z*z)
        acceleration_r = -grav_const_km*sbh_menc(d)/(d*d)
        vx = x/d*acceleration_r
        vy = y/d*acceleration_r
        vz = z/d*acceleration_r
    end subroutine dm_accel_sbh_only

    subroutine dm_setup()
        ! use triaxpotent, only: tp_setup
        real(kind=dp) :: darkmass, dm_zeta, zh_betai, tmp_gamma, zh_gammln
        ! call tp_setup()

        select case (dm_profile_type)
        case (0)
            print *, 'No additional DM halo'
        case (1)
            !             dmparam(1) = concentration
            !        dmparam(2) = dm_fraction   (fraction of DM mass within R200 radius)
            if (n_dmparam .ne. 2) stop 'wrong number of NFW halo parameters'
            print *, 'Parameters of NFW concentration and fraction', dmparam(1), dmparam(2)

            ! Parameters for NFW profile
            rhoc = (200.0_dp/3.0_dp)*rho_crit*dmparam(1)**3/ &
                   (log(1.0_dp + dmparam(1)) - dmparam(1)/(1.0_dp + dmparam(1)))

            rc = (3.0_dp/(800.0_dp*pi_d*rho_crit*dmparam(1)**3) &
                  *dmparam(2)*totalmass)**(1.0_dp/3.0_dp)

            ! 12 Oct 2011: LW found unit conversion bug in print statment
            print *, "Parameters of NFW potential (rho_c in solarmass/km^3 and r_c in km): ", rhoc, rc
            ! Calculate M200, in Msun
            darkmass = 800_dp*pi_d/3_dp*rho_crit*rc**3*dmparam(1)**3

            print *, "Total stellar mass is (Msun): ", totalmass
            print *, "Total dark halo mass (M200 in Msun): ", darkmass

        case (2)
            !             dmparam(1) = rhoc
            !        dmparam(2) = rc
            if (n_dmparam .ne. 2) stop 'wrong number of Hernquist halo parameters'
            rhoc = dmparam(1)
            rc = dmparam(2)
            print *, 'Parameters of Hernquist profile', rhoc, rc
        case (3)
            print *, "  * triaxial cored logarithmic potential. "
            ! from Thomas et al. 2005  & B&T 1987 (p. 46)
            if (n_dmparam .ne. 4) stop 'wrong number of halo parameters'

            print *, "  Vc (km/s), Rc (kpc,km):", dmparam(1), dmparam(2), dmparam(2)*parsec_km*1d3
            if (dmparam(1) .le. 0.0_dp) stop 'VC < 0'
            if (dmparam(2) .le. 0.0_dp) stop 'Rc < 0'

            print *, "  flattening p & q", dmparam(3), dmparam(4)
            if (dmparam(3) .gt. 1.0_dp .or. dmparam(4) .le. 0.0_dp &
                .or. dmparam(3) .lt. dmparam(4)) stop ' Flattening is not 0<q<=p<=1'

            ! turning p and q into p^2 and q^2
            dmparam(3) = dmparam(3)**2
            dmparam(4) = dmparam(4)**2

            ! Turning Core radius from kpc to km^2
            dmparam(2) = (dmparam(2)*parsec_km*1d3)**2.0_dp

            ! Turning VC into km^2
            dmparam(1) = dmparam(1)**2.0_dp

            !case (4)
            !read (unit=13, fmt=*) dm_profile_rhoc, dm_profile_parameter, r200, dm_rho_crit, dm_profile_a, dm_profile_b, dm_profile_c
            !print*,'Parameters of NFW(triaxial approximation)', dm_profile_rhoc, dm_profile_parameter, r200, dm_rho_crit
            !print*,'rho crit:', rho_crit
            !print*, dm_profile_a, dm_profile_b, dm_profile_c
            !dm_profile_axes_length = dm_profile_a**2 + dm_profile_b**2 + dm_profile_c**2
            !print*,'unnormalized axes length:', dm_profile_axes_length
            !dm_profile_a = dm_profile_a *sqrt(3/dm_profile_axes_length)
            !dm_profile_b = dm_profile_b *sqrt(3/dm_profile_axes_length)
            !dm_profile_c = dm_profile_c *sqrt(3/dm_profile_axes_length)
            !dm_profile_axes_length = dm_profile_a**2 + dm_profile_b**2 + dm_profile_c**2
            !print*,' normalized axes length:', dm_profile_axes_length
        case (5)
            !             dmparam(1) = concentration=r_vir/r_s
            !        dmparam(2) = dm_fraction   (fraction of DM mass within R200 radius)
            if (n_dmparam .ne. 3) stop 'wrong number of gNFW halo parameters'
            print *, 'Parameters of NFW concentration, Mvir, inner log-slope', dmparam(1), dmparam(2), dmparam(3)
            !Update, JJA. Use c_vir=r_vir/r_s, M_vir=(4/3)*pi*Del_c*rho_crit*r_v^3,
            !analytically integrated to M(r), and equate to get rho_s/rho_crit==delta_c.
            !dm_zeta is a simplification for Barnable+12 Eq 10.
            if (dmparam(3) .lt. 1.0_dp) then
                dm_zeta = ((1.0_dp + dmparam(1))/dmparam(1))**(dmparam(3) - 2.0_dp) &
                          *(2.0_dp*dmparam(3)*dmparam(1) - 3.0_dp &
                            *dmparam(1) + dmparam(3) - 2.0_dp)/ &
                          (dmparam(3)*dmparam(3) - 3.0_dp*dmparam(3) + 2.0_dp) &
                          /dmparam(1) + exp(zh_gammln(2.0_dp - dmparam(3)) - zh_gammln(1.0_dp - dmparam(3))) &
                          *zh_betai(1.0_dp - dmparam(3), 0.0_dp, dmparam(1) &
                                    /(dmparam(1) + 1.0_dp))/(1.0_dp - dmparam(3))
            else if (dmparam(3) .eq. 1.0_dp) then
                dm_zeta = log(1.0_dp + dmparam(1)) - dmparam(1)/(1.0_dp + dmparam(1))
            else
                !Must protect against the argument in gammln going negative, using Euler's
                !reflection formula. Will be singular if gamma runs away to large integers >1,
                !but those cases are unreasonable anyway.
                tmp_gamma = pi_d/sin(pi_d*(1.0_dp - dmparam(3)))/exp(zh_gammln(dmparam(3)))
                dm_zeta = ((1.0_dp + dmparam(1))/dmparam(1))**(dmparam(3) - 2.0_dp) &
                          *(2.0_dp*dmparam(3)*dmparam(1) - 3.0_dp &
                            *dmparam(1) + dmparam(3) - 2.0_dp)/ &
                          (dmparam(3)*dmparam(3) - 3.0_dp*dmparam(3) + 2.0_dp) &
                          /dmparam(1) + (exp(zh_gammln(2.0_dp - dmparam(3)))/tmp_gamma) &
                          *zh_betai(1.0_dp - dmparam(3), 0.0_dp, dmparam(1)/(dmparam(1) + 1.0_dp)) &
                          /(1.0_dp - dmparam(3))
            end if

            rhoc = (200.0_dp/3.0_dp)*rho_crit*dmparam(1)**3/dm_zeta

            rc = (3.0_dp*dmparam(2)/(800.0_dp*pi_d*rho_crit*dmparam(1)**3))**(1.0_dp/3.0_dp)
            gamma_var = dmparam(3)
            ! 12 Oct 2011: LW found unit conversion bug in print statment
            print *, "Parameters of NFW potential (rho_c in solarmass/km^3 and r_c in km): ", rhoc, rc
            ! Calculate M200, in Msun
            darkmass = dmparam(2)
            ! print*, rhoc, rc, "DM values----------------------------"

        end select

        ! Optional sBH subcluster, independent of the halo slot above.
        ! sbhparam is unallocated when the block is absent, so every
        ! access is gated on sbh_profile_type.
        sbh_present = (sbh_profile_type .eq. 6)
        if (sbh_present) then
            call sbh_assign()
            print *, '  * sBH subcluster: rho0, a, alpha, beta, gamma =', &
                sbh_rho0, sbh_a, sbh_al, sbh_be, sbh_ga
            print *, '    total sBH mass (Msun):', sbh_mtot()
        end if

    end subroutine dm_setup

    subroutine dm_stop()
        ! use triaxpotent, only : tp_stop
        ! call tp_stop()  ! function does not exist, but should.
    end subroutine dm_stop

    subroutine dm_potent(x, y, z, pot)
        use triaxpotent, only: tp_potent
        use initial_parameters
        real(kind=dp), intent(in) ::  x, y, z
        real(kind=dp), intent(out):: pot
        real(kind=dp) :: d, d2, dnorm, xi, ibeta_v2, ibeta_v3, zh_betai, sbh_yv

        d2 = x*x + y*y + z*z

        call tp_potent(x, y, z, pot)

        ! add Plummer style black hole
        pot = pot + grav_const_km*xmbh/sqrt(d2 + softl_km*softl_km)

        select case (dm_profile_type)
        case (0)
            !blank
        case (1)
            ! add NFW dark halo
            !d =sqrt(d2)
            if (sqrt(d2)/rc .ge. 1.0) then
                pot = pot + 4.0_dp*pi_d*grav_const_km*rhoc*rc**3/sqrt(d2)*log(1.0_dp + sqrt(d2)/rc)
            else
                ! indentity log (1+x) = 2* atanh(x/(2+x)) , required when 0<x<<1
                pot = pot + 4.0_dp*pi_d*grav_const_km*rhoc*rc**3/sqrt(d2)*2*atanh((sqrt(d2)/rc)/(2 + sqrt(d2)/rc))
            end if
        case (2)
            ! Hernquist
            d = sqrt(d2)
            pot = pot + 4.0_dp*pi_d*grav_const_km*rhoc*rc**2/(2*(1 + d/rc))
        case (3)
            ! cored logarihtmic
            ! phi=1/2*vc^2*log(rc^2+x^2+y^2/p^2+z^2/q^2)
            pot = pot - 0.5_dp*dmparam(1)*log(dmparam(2) + (x**2.0_dp + y**2.0_dp/dmparam(3) + z**2.0_dp/dmparam(4)))
            if ((x**2.0_dp + y**2.0_dp/dmparam(3) + z**2.0_dp/dmparam(4))/dmparam(2) .le. 1.0d-14) &
                stop ' potential fails log(x+y) test'
            ! density rho is the laplacian of the potential phi:
            !rho = -vc**2*(-p**4*q**4*rc**2+x**2*p**4*q**4-p**2*q**4*y**2-p*        $
            !    *4*q**2*z**2-q**4*rc**2*p**2-q**4*x**2*p**2+y**2*q**4-q**2*z**2*   $
            !    p**2-p**4*rc**2*q**2-p**4*x**2*q**2-p**2*y**2*q**2+z**2*p**4)      $
            !    /(rc**2*p**2*q**2+x**2*p**2*q**2+y**2*q**2+z**2*p**2)**2

            !case (4)
            ! r = d
            ! ra = dm_profile_parameter
            ! re = sqrt(x**2/dm_profile_a**2 + y**2/dm_profile_b**2 + z**2/dm_profile_c**2)
            ! rtilde = (ra+r)*re / (ra+re)
            !
            ! pot = pot + 4.0_dp * pi_d * grav_const_km*dm_profile_rhoc*dm_profile_parameter**3/rtilde * log(1.0_dp + rtilde/dm_profile_parameter)
        case (5)
            !This can be derived from Zhao'96 Eq 6-7.
            dnorm = sqrt(d2)/rc
            xi = dnorm/(1.0_dp + dnorm)
            ibeta_v2 = zh_betai(3.0_dp - gamma_var, 0.0_dp, xi)
            ibeta_v3 = zh_betai(1.0_dp, 2.0_dp - gamma_var, 1.0_dp - xi)
            pot = pot + (4.0_dp*pi_d)*grav_const_km*rhoc*(ibeta_v2/dnorm &
                                                          + ibeta_v3)*rc*rc
        end select

        if (sbh_present) then
            d = sqrt(d2)
            sbh_yv = sbh_yvar(d)
            ! this module's `pot` is positive (psi = -Phi), matching the
            ! Plummer and NFW terms above
            pot = pot + grav_const_km &
                  *(sbh_menc(d, sbh_yv)/d + sbh_outer_tail(d, sbh_yv))
        end if

    end subroutine dm_potent

!+++++++++++++++++++++++++++++++++++
    subroutine dm_accel(x, y, z, vx, vy, vz)
        use triaxpotent, only: tp_accel
        use initial_parameters
        real(kind=dp), intent(in) ::  x, y, z
        real(kind=dp), intent(out):: vx, vy, vz
        !------------------------------------
        real(kind=dp) :: t, t1, t2, t3, t4, d2, acceleration_r
        real(kind=dp) :: d, ibeta_v1, ibeta_v2, ibeta_v3, xi, dnorm, zh_betai, zh_beta
        !  integer(kind=i4b) :: p,r

        call tp_accel(x, y, z, vx, vy, vz)

        d2 = x*x + y*y + z*z
        ! Add Plummer style blackhole.
        t = -grav_const_km*xmbh*(d2 + softl_km*softl_km)**(-3.0_dp/2.0)
        vx = vx + x*t
        vy = vy + y*t
        vz = vz + z*t

        select case (dm_profile_type)
        case (0)
            !blank
        case (1)   ! Add NFW dark matter halo
            t1 = -4.0_dp*pi_d*grav_const_km*rhoc*rc**3/d2
            ! indentity log (1+x) = 2* atanh(x/(2+x)) , required when 0<x<<1
            if (sqrt(d2)/rc .ge. 1.0) then
                t2 = log(1.0_dp + sqrt(d2)/rc)
            else
                t2 = 2*atanh((sqrt(d2)/rc)/(2 + sqrt(d2)/rc))
            end if
            t3 = (sqrt(d2)/rc)/(1.0_dp + sqrt(d2)/rc)
            vx = vx + x/sqrt(d2)*t1*(t2 - t3)
            vy = vy + y/sqrt(d2)*t1*(t2 - t3)
            vz = vz + z/sqrt(d2)*t1*(t2 - t3)
            if (x/sqrt(d2)*t1*(t2 - t3) .gt. 0 .and. x .gt. 0) then
                print *, vx, x/sqrt(d2)*t1*(t2 - t3), d2
                print *, t1, t2, t3
                print *, sqrt(d2)/rc, d2, rc
                stop 'NFW accelations flipped sign'
            end if
        case (2)     !Hernquist
            acceleration_r = -2.0_dp*pi_d*grav_const_km*rhoc*rc/(1 + sqrt(d2)/rc)**2
            vx = vx + x/sqrt(d2)*acceleration_r
            vy = vy + y/sqrt(d2)*acceleration_r
            vz = vz + z/sqrt(d2)*acceleration_r
        case (3) ! cored logarihtmic
            ! phi=1/2*vc^2*log(rc^2+x^2+y^2/p^2+z^2/q^2)
            ! vx= diff(phi,x)
            vx = vx - dmparam(1)*x/(dmparam(2) + x*x + y*y/dmparam(3) + z*z/dmparam(4))
            vy = vy - dmparam(1)*(y/dmparam(3))/(dmparam(2) + x*x + y*y/dmparam(3) + z*z/dmparam(4))
            vz = vz - dmparam(1)*(z/dmparam(4))/(dmparam(2) + x*x + y*y/dmparam(3) + z*z/dmparam(4))

            !case (4)     !Vogelsberger
            !            d = sqrt(d2)
            !                ! see Vogelsberger+ 2008
            !                r = d
            !                ra = dm_profile_parameter
            !                re = sqrt(x**2/dm_profile_a**2 + y**2/dm_profile_b**2 + z**2/dm_profile_c**2)
            !                rtilde = (ra+r)*re / (ra+re)
            !
            !                t1 = -4.0_dp*pi_d*grav_const_km*dm_profile_rhoc*dm_profile_parameter**3/rtilde**2
            !                t2 = log(1.0_dp + rtilde/dm_profile_parameter)
            !                t3 = (rtilde/dm_profile_parameter)/(1.0_dp+rtilde/dm_profile_parameter)
            !
            !                acceleration_rtilde = t1*(t2 - t3)
            !
            !                drtildedx = 1 / (ra+re)**2 * (1/re * x / dm_profile_a**2 * (ra+r) * ra + 1/r * x * (ra+re)*re)
            !                drtildedy = 1 / (ra+re)**2 * (1/re * y / dm_profile_b**2 * (ra+r) * ra + 1/r * y * (ra+re)*re)
            !                drtildedz = 1 / (ra+re)**2 * (1/re * z / dm_profile_c**2 * (ra+r) * ra + 1/r * z * (ra+re)*re)
            !                vx = vx + acceleration_rtilde * drtildedx
            !                vy = vy + acceleration_rtilde * drtildedy
            !                vz = vz + acceleration_rtilde * drtildedz
        case (5) ! gNFW
            !Derived by JJA, starting with the results of Zhao'06 Eq 6-7.
            dnorm = sqrt(d2)/rc
            xi = dnorm/(1.0_dp + dnorm)
            ibeta_v2 = zh_betai(3.0_dp - gamma_var, 0.0_dp, xi)
            acceleration_r = 4.0_dp*pi_d*grav_const_km*rhoc*rc/dnorm
            t1 = xi**(2.0_dp - gamma_var)/(1.0_dp - xi)/rc/dnorm/(1.0_dp + dnorm)**2
            t2 = xi**(1.0_dp - gamma_var)/rc/(1.0_dp + dnorm)**2
            t3 = ibeta_v2*rc/d2
            t4 = acceleration_r*(t1 - t2 - t3)
            vx = vx + x*t4
            vy = vy + y*t4
            vz = vz + z*t4
        end select

        if (sbh_present) then
            d = sqrt(d2)
            ! exact for all gamma < 3
            acceleration_r = -grav_const_km*sbh_menc(d)/d2
            vx = vx + x/d*acceleration_r
            vy = vy + y/d*acceleration_r
            vz = vz + z/d*acceleration_r
        end if

    end subroutine dm_accel
end module dmpotent
