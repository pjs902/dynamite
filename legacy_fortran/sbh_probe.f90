! Standalone probe: prints the sBH potential and radial acceleration on a
! log grid, for cross-checking against the Python implementation (Task 7).
! Usage:  ./sbh_probe <rho0> <a> <alpha> <beta> <gamma>
!
! Calls the *_sbh_only wrappers added in Task 6 (module dmpotent), which
! share sbh_assign/sbh_menc/sbh_outer_tail with the real dm_setup/dm_potent/
! dm_accel code paths, so nothing here duplicates the physics.
program sbh_probe
    use numeric_kinds
    use initial_parameters
    use dmpotent
    implicit none
    character(len=64) :: arg
    real(kind=dp) :: r, pot, vx, vy, vz
    integer(kind=i4b) :: i

    sbh_profile_type = 6
    n_sbhparam = 5
    allocate (sbhparam(5))
    do i = 1, 5
        call get_command_argument(i, arg)
        read (arg, *) sbhparam(i)
    end do
    call dm_setup_sbh_only()

    ! r/a = 1e-6 .. 1e4, log-spaced (a = sbhparam(2))
    do i = -60, 40
        r = sbhparam(2)*10.0_dp**(real(i, dp)/10.0_dp)
        call dm_potent_sbh_only(r, 0.0_dp, 0.0_dp, pot)
        call dm_accel_sbh_only(r, 0.0_dp, 0.0_dp, vx, vy, vz)
        write (*, '(3ES30.18)') r, pot, vx
    end do
end program sbh_probe
