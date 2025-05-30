CPPCMD		 = /usr/bin/cpp
CPPINCLUDEDIR	 = 
CPPOTHERFLAGS	 = -P -C -nostdinc -nostdinc++
CPP		 = $(CPP) -I$(CPPINCLUDEDIR) $(CPPOTHERFLAGS)

UMAKE		 = ${CUTER}/build/scripts/umake

CMP		 = gfo	
MACHINE		 = Intel-like-PC	
MCH		 = pc	
COMPILE		 = gfortran -c -ffixed-form -fno-second-underscore	
LOAD		 = gfortran -ffixed-form -fno-second-underscore		
ISF9095		 = yes	
COMPILE9095	 = gfortran -c -fno-second-underscore	
LOAD9095	 = gfortran -fno-second-underscore	
FFLAGS		 = -O	
SPECIALFLAGS	 = 	
NBYTES		 = 8	
CCOMPILE	 = 
CLOAD		 = CLoadCmd
CFLAGS		 = CFlags
SPECIALLIBS = SpecialLibs
R1		 = 1.1920930E-07
R2		 = 5.9604646E-08
R3		 = 1.1754945E-38
R4		 = 1.1754945E-38
R5		 = 3.4028234E+38
D1		 = 2.2204460492503132D-16
D2		 = 1.1102230246251566D-16
D3		 = 2.225073858507202D-308
D4		 = 2.225073858507202D-308
D5		 = 1.797693134862314D+308

PRECISION = double
PRC = CD
PRC90 = \!D

SIZE = large
SIZ = CBIG
SIZ90 = \!BIG

PATHSEP = /
SHELL = /bin/sh

TOP = ../..
CURRENT_DIR = /double/lib

UMAKE_DEFINES = 
UMAKE_CONFIGS = -C$(MYCUTER)/config 

UMAKE_CPP_FLAGS = -Ptraditional -Pw

IRULESRC = $(MYCUTER)/config	
UMAKE_CMD = $(UMAKE) -I. -I$(IRULESRC) $(UMAKE_DEFINES) $(UMAKE_CONFIGS) $(UMAKE_CPP_FLAGS)

TMP = /tmp

SYSTEM = Linux
KEYSYS = CUNIX

MKDIR = /bin/mkdir
CP = /bin/cp
RM = /bin/rm -f
MV = /bin/mv
CAT = /bin/cat
CHMOD = /bin/chmod
SED = /bin/sed
LN = /bin/ln
LS = /bin/ls
AR = /usr/bin/ar
ARFLAGS= -cr
RMDIR = /bin/rmdir
GREP = /bin/grep
AWK = /usr/bin/gawk
HEAD = /usr/bin/head
TAIL = /usr/bin/tail
WC = /usr/bin/wc
MAKE = /usr/bin/make
LD = /usr/bin/ld

MEXFORTRAN = /usr/bin/mex 
MEXFFLAGS = -fortran

INSTALL = install

COMMANDLIST = CMP MACHINE MCH COMPILE LOAD COMPILE9095 LOAD9095 FFLAGS NBYTES CCOMPILE CLOAD CFLAGS SPECIALLIBS R1 R2 R3 R4 R5 D1 D2 D3 D4 D5 PRECISION PRC PRC90 SIZE SIZ SIZ90 SYSTEM KEYSYS MKDIR CP RM MV CAT CHMOD SED LN LS AR ARFLAGS RMDIR GREP AWK HEAD TAIL WC MAKE LD MEXFORTRAN MEXFFLAGS

.SUFFIXES: .f .c .a

DESTDIR =	

INTERFACESSRC = $(CUTER)/build/prototypes
INTCONFIG = $(MYCUTER)/$(PRECISION)/config
TOOLSSRCDIR = $(CUTER)/common/src/tools
TOOLSSRCDIRSLASH = $(CUTER)/common/src/tools/
TOOLSLIBDIR = $(MYCUTER)/$(PRECISION)/lib

VPATH = $(TOOLSSRCDIR)

local.o:
	cd $(MYCUTER)/$(PRECISION)/config; 	$(MAKE) $(MFLAGS) local.o; 	cd $(MYCUTER)/$(PRECISION)/lib;

.f.a:
	@printf ' %-12s %-15s\t\t' "Building" $%
	$(SED) "/#{sizing}/r $(INTCONFIG)/tools.siz" $(TOOLSSRCDIR)/${%:.o=.f} | $(SED) -f $(INTCONFIG)/cast.sed > $(TOOLSLIBDIR)/${%:.o=.f}
	$(COMPILE) $(FFLAGS) -o $(TOOLSLIBDIR)/$% $(TOOLSLIBDIR)/${%:.o=.f}
	$(AR) -crs $@ $(TOOLSLIBDIR)/$%
	@printf '[Ok]\n'
	$(RM) $(TOOLSLIBDIR)/$%

.f.o:
	@printf ' %-12s %-15s\t\t' "Building" ${@}
	$(SED) "/#{sizing}/r $(INTCONFIG)/tools.siz" $(TOOLSSRCDIR)/${@:.o=.f} | $(SED) -f $(INTCONFIG)/cast.sed > $(TOOLSLIBDIR)/${@:.o=.f}
	$(COMPILE) $(FFLAGS) -o ${@} ${@:.o=.f}
	@printf '[Ok]\n'

.c.a: $(CUTER)/common/include/cuter.h
	@printf ' %-12s %-15s\t\t' "Building" $%
	$(CCOMPILE) $(CFLAGS) -I$(CUTER)/common/include -o $(TOOLSLIBDIR)/$% $(TOOLSSRCDIR)/${%:.o=.c}
	$(AR) -crs $@ $(TOOLSLIBDIR)/$%
	@printf '[Ok]\n'

.c.o:
	@printf ' %-12s %-15s\t\t' "Building" ${@}
	$(CCOMPILE) $(CFLAGS) -I$(CUTER)/common/include -o ${@} $(TOOLSSRCDIR)/${@:.o=.c}
	@printf '[Ok]\n'

all::

baseconfig:
	cd $(MYCUTER)/$(PRECISION)/config; 	$(MAKE) $(MFLAGS) all;

all:: baseconfig

.: $(INTCONFIG)/tools.siz $(INTCONFIG)/cast.sed
	touch $(TOOLSSRCDIRSLASH)*.f

all:: .

C_O_FILES =
CLIBFILES =

DotOs =	asmbe.o asmbl.o ccfg.o 	ccfsg.o ccifg.o ccifsg.o 	cdh.o cdimen.o cdimse.o 	cdimsh.o cdimsj.o ceh.o 	cfn.o cgr.o cgrdh.o 	cidh.o cish.o cjprod.o cnames.o 	cofg.o cprod.o creprt.o 	cscfg.o cscifg.o csetup.o 	csgr.o csgreh.o csgrsh.o 	csh.o csh1.o cvarty.o elgrd.o connames.o hsprd.o initw.o others.o pbname.o ubandh.o udh.o udimen.o udimse.o udimsh.o ueh.o ufn.o ugr.o ugrdh.o ugreh.o ugrsh.o unames.o uofg.o uprod.o ureprt.o usetup.o ush.o uvarty.o varnames.o ${C_O_FILES}

libcuter.so: local.o $(DotOs)
	$(LD) -G $(DotOs) $(MYCUTER)/$(PRECISION)/config/local.o -o $@

libcuter.a: 	libcuter.a(asmbe.o) libcuter.a(asmbl.o) libcuter.a(ccfg.o) 	libcuter.a(ccfsg.o) libcuter.a(ccifg.o) libcuter.a(ccifsg.o) 	libcuter.a(cdh.o) libcuter.a(cdimen.o) libcuter.a(cdimse.o) 	libcuter.a(cdimsh.o) libcuter.a(cdimsj.o) libcuter.a(ceh.o) 	libcuter.a(cfn.o) libcuter.a(cgr.o) libcuter.a(cgrdh.o) 	libcuter.a(cidh.o) libcuter.a(cish.o) libcuter.a(cjprod.o) libcuter.a(cnames.o) 	libcuter.a(cofg.o) libcuter.a(cprod.o) libcuter.a(creprt.o) 	libcuter.a(cscfg.o) libcuter.a(cscifg.o) libcuter.a(csetup.o) 	libcuter.a(csgr.o) libcuter.a(csgreh.o) libcuter.a(csgrsh.o) 	libcuter.a(csh.o) libcuter.a(csh1.o) libcuter.a(cvarty.o) libcuter.a(elgrd.o) libcuter.a(connames.o) libcuter.a(hsprd.o) libcuter.a(initw.o) libcuter.a(others.o) libcuter.a(pbname.o) libcuter.a(ubandh.o) libcuter.a(udh.o) libcuter.a(udimen.o) libcuter.a(udimse.o) libcuter.a(udimsh.o) libcuter.a(ueh.o) libcuter.a(ufn.o) libcuter.a(ugr.o) libcuter.a(ugrdh.o) libcuter.a(ugreh.o) libcuter.a(ugrsh.o) libcuter.a(unames.o) libcuter.a(uofg.o) libcuter.a(uprod.o) libcuter.a(ureprt.o) libcuter.a(usetup.o) libcuter.a(ush.o) libcuter.a(uvarty.o) libcuter.a(varnames.o) 	${CLIBFILES}
	$(AR) -crs libcuter.a $(MYCUTER)/$(PRECISION)/config/local.o
	$(RM) *.f *.c

cleanlib:
	$(RM) $(DotOs) *.f *.c

all:: local.o
all:: libcuter.a

all:: cleanlib

clean:: 
	$(RM) *.CKP *.ln *.BAK *.bak core errs ,* *~ tags TAGS make.log MakeOut "#"*

purge:: 
	$(RM) libcuter.a

purge:: 
	$(RM) libcuter.so

purge:: 
	$(RM) *.f *.o

MFLAGS =

emptyrule::

clean:: 
	$(RM) *.CKP *.ln *.BAK *.bak core errs ,* *~ tags TAGS make.log MakeOut "#"*

Makefile:: 
	-@if [ -f Makefile ]; then set -x; \
	$(RM) Makefile.bak; $(MV) Makefile Makefile.bak; \
	else exit 0; fi 
	$(UMAKE_CMD) -DTOPDIR=$(TOP) -DCURDIR=$(CURRENT_DIR) -Dlinux -DIsgfo -DLargeSize -DDoublePrecision -DIsa90Compiler -DNoCcomp -Ptraditional -Pw

Makefiles::

includes::

