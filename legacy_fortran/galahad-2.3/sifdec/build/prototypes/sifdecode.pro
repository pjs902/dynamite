#!/bin/csh -f
# sifdecode: script to decode a sif file
#  ( Last modified on 23 Dec 2000 at 17:29:56 )
#
#{version}
#
# Use: sifdecode [-s] [-h] [-k] [-o j] [-l secs] [-f] [-b] [-a j] [-show] [-param name=value[,name=value...]] [-force] [-debug] probname[.SIF]
#
# where: options -s     : run the single precision version
#                         (Default: run the double precision version)
#                -h     : print this help and stop execution
#                -k     : keep the load module after use
#                         (Default: delete the load module)
#                -o     : 0 for silent mode, 1 for brief description of
#                         the stages executed.
#                         (Default: -o 0)
#                -l     : sets a limit of secs second on the runtime
#                         (Default: 99999999 seconds)
#                -f     : use automatic differentiation in Forward mode
#                -b     : use automatic differentiation in Backward mode
#                -a     : 1 use the older HSL automatic differentiation package AD01
#                         2 use the newer HSL automatic differentiation package AD02
#                         (Default: -a 2)
#                -show  : displays possible parameter settings for
#                         probname[.SIF]. Other options are ignored
#                -param : cast probname[.SIF] against explicit parameter
#                         settings. Several parameter settings may be
#                         given as a comma-separated list following
#                         -param or using several -param flags.
#                         Use -show to view possible settings.
#                -force : forces setting of a parameter to the given value,
#                         even if this value is not specified in the file.
#                         This option should be used with care.
#                         (Default: do not enforce).
#                -debug : links all the libraries, creates the executable
#                         and stop to allow debugging. This option
#                         automatically disables -n and enables -k.
#
#       probname      probname.SIF is the name of the file containing
#                     the SIF file for the problem of interest.
#

#
#  N. Gould, D. Orban & Ph. Toint, November 7th, 2000
#

#{cmds}

unset noclobber

#
# Environment check
#

@ abort = 0

if( ! $?SIFDEC ) then
    echo ' SIFDEC is not set.'
    echo ' It should point to the directory where you installed SifDec.'
    echo ' Set it to the appropriate value and re-run.'
    @ abort = 1
endif

if( ! $?MYSIFDEC ) then
    echo ' MYSIFDEC is not set.'
    echo ' It should point to your working instance of SifDec.'
    echo ' Set it to the appropriate value and re-run.'
    echo ''
    @ abort = 1
endif

if( $abort ) then
    echo ' Aborting.'
    exit 7
endif

#
#  variables for each option
#

#
#  LIMIT (maximum cputime)
#

set LIMIT=99999999

#
# PRECISION = single (single precision), = double (double precision)
#

set PRECISION="double"
set prec = 1

#
# KEEP = (null) (discard load module after use), = "-k" (keep it)
#

setenv KEEP ""

#
# DEBUGARG = "" (normal execution)
#

set DEBUGARG = ""

#
# OUTPUT = 0 (summary output), = 1 (detailed output from decoder)
#

set OUTPUT=0

#
#   automatic = 0 (provided), = 1 (automatic forward), = 2 (automatic backward)
#

set automatic = 0

#
#   ad0 = 1 (AD01 used), = 2 (AD02 used)
#

set ad0 = 2

#
# FORCE = 0 (do not enforce -param settings), = 1 (enforce settings)
#

set FORCE=0

#
#  interpret arguments
#

set METHOD=3

#
#  Check whether variable PAC has been set;
#  this allows sifdecode to be called from the command line.
#

if( ! $?PAC ) then
    setenv PAC sifdecode
    setenv caller sifdecode
    set thiscaller = `echo $0:t`
else
    set thiscaller = sd${PAC}
endif

@ last = $#argv

if($last == 0) then
  echo "Use: $thiscaller [-s] [-h] [-k] [-o j] [-l secs] [-f] [-b] [-a j] [-show] [-param name=value[,name=value...]] [-debug] probname[.SIF]"
  exit 1
endif

@ i=1
@ nbparam=0   # counts the number of parameters passed using -param
set PARAMLIST = ""

while ($i <= $last)
  set opt=$argv[$i]
  if ("$opt" == '-h' || "$opt" == '--help') then
    $MYSIFDEC/bin/helpmsg
    exit 0
  else if ("$opt" == '-s') then
    set PRECISION="single"
    set prec = 0
  else if ("$opt" == '-k') then
    set KEEP="-k"
  else if ("$opt" == '-o') then
    @ i++
    set OUTPUT=$argv[$i]
  else if ("$opt" == '-l') then
    @ i++
    set LIMIT=$argv[$i]
  else if ("$opt" == '-m') then
    @ i++
    set METHOD = $argv[$i]
  else if ("$opt" == '-f') then
    set automatic = 1
  else if ("$opt" == '-b') then
    set automatic = 2
  else if ("$opt" == '-a') then
    @ i++
    if ( $argv[$i] == '1' || $argv[$i] == '2' ) then
	set ad0 = $argv[$i]
    else
	echo " ${thiscaller} : error processing -a flag"
	exit 6
    endif
  else if ("$opt" == '-param') then
    # parameters can either be separated by -param flags
    # or by commas within a -param flag, e.g.:
    # -options... -param par1=val1,par2=val2 -otheroptions... -param par3=val3
    @ i++
    # see if some parameters were given as a comma-separated list
    set nbparInThisGroup = `echo $argv[$i] | $AWK -F, '{print NF}'`
    if ($nbparInThisGroup == 0) then
	echo " ${thiscaller} : error processing -param flag"
	exit 5
    endif
    # check if what follows -param looks like a parameter setting
    set chkparam = `echo $argv[$i] | $GREP =`
    if ( $chkparam == "" ) then
    echo " ${thiscaller} : error processing -param flag"
	exit 5
    endif
    # store the parameters in the PARAM array
    set PARAMLIST="$PARAMLIST `echo $argv[$i] | $SED -e 's/,/ /g'`"
    @ nbparam = `expr ${nbparam} + ${nbparInThisGroup}`
  else if ("$opt" == '-show') then
    set SHOWPARAMS=1
    @ i=`expr ${last} + 1`                      # skip remaining options
  else if ("$opt" == '-force') then
    set FORCE=1
  else if ("$opt" == '-debug') then
    set KEEP="-k"
    set DEBUGARG = "-debug"
  endif
  @ i++
end

# Create problem names

@ last = $#argv
set PROBLEM  = $argv[$last]
set PROBNAME = `echo $PROBLEM | $AWK -F/ '{print $NF}' | $AWK -F. '{print $1}'`
set EXT = `echo $PROBLEM | $AWK -F/ '{print $NF}' | $AWK -F. '{print $2}'`

set PROBDIR = `echo $PROBLEM | $AWK 'BEGIN{FS="\n"}{ l = split( $1, farray, "/" ); head = ""; if( l>1 ) { head = farray[1]; for ( i=2; i<l; i++ ) { head = head "/" farray[i]; } } print head;}'`

set probDirNotGiven = "false"

if ( "$PROBDIR" == "" ) then
    set probDirNotGiven = "true"
    set PROBDIR = '.'
endif


set problemLoc = ${PROBDIR}/${PROBNAME}.SIF

# Specify correct path for problemLoc

set lookInMastSif = 0
if (! -e "$problemLoc" && "$probDirNotGiven" == "true" && $?MASTSIF) then
    set lookInMastSif = 1
    set problemLoc = ${MASTSIF}/${PROBNAME}.SIF
endif

# See whether the specified SIF file exists

if (! -e "$problemLoc") then
  if (! $?MASTSIF) then
    echo ' MASTSIF is not set'
  endif
  echo "file $PROBNAME.SIF is not known in directories"
  if ( $PROBDIR == "$PROBLEM") set PROBDIR = $PWD
  echo " $PROBDIR or "'$MASTSIF'
  echo "possible choices are:"
  echo ' '
  cd $PROBDIR; $LS -x *.SIF
  echo ' '
  exit 2
endif

# See if the -show flag is present

if ( $?SHOWPARAMS ) then
    set SHOWDOTAWK = ${MYSIFDEC}/bin/show.awk
    if ($OUTPUT) then
	echo "possible parameter settings for $PROBNAME are:"
    endif
    $GREP '$-PARAMETER' ${problemLoc} | $AWK -f $SHOWDOTAWK
    exit 8
# Note: the 'exit 8' command is used if -show is given on the command-line
# of a sd* interface. If the exit code were 0, the interface would go on,
# invoking the solver it interfaces.
endif

# Check if -param arguments have been passed and process them

if ( $nbparam > 0 ) then
    # part parameter names from their value
    @ p = 1
    set PARLIST = `echo $PARAMLIST:q`
    set PARNAMELIST  = `echo $PARAMLIST:q`
    set PARVALUELIST = `echo $PARAMLIST:q`
    while ( $p <= $nbparam )
	set PARNAMELIST[$p]  = `echo $PARLIST[$p] | $AWK -F= '{print $1}'`
	set PARVALUELIST[$p] = `echo $PARLIST[$p] | $AWK -F= '{print $2}'`
	@ p++
    end

    set PARAMDOTAWK = ${MYSIFDEC}/bin/param.awk
    # substitute the chosen parameter values in the SIF file
    @ p = 1

    # the easiest looks like creating a sed script which operates
    # all the necessary changes at once. We fill this sed script
    # as each parameter setting is examined in turn.

    # i give it a name depending on the current pid
    # hopefully, this is a unique name.

    set sedScript = $TMP/$$_${PROBNAME}.sed
    set sifFile   = $TMP/$$_${PROBNAME}.SIF
    echo '' > $sedScript

    while ( $p <= $nbparam )
	# see if parameter number p can be set to value number p
	# if so, retrieve the number of the matching line in the file
	# and the numbers of the lines which should be commented out.
	set matchingLines = "0"
	set nonMatchingLines = "0"

	set matchingLines = `$GREP -n '$-PARAMETER' ${problemLoc} | $AWK -F'$' '{print $1}' | $AWK -v pname=$PARNAMELIST[$p] -v pval=$PARVALUELIST[$p] -v doesmatch=1 -f ${PARAMDOTAWK}`

	set nonMatchingLines = `$GREP -n '$-PARAMETER' ${problemLoc} | $AWK -F'$' '{print $1}' | $AWK -v pname=$PARNAMELIST[$p] -v pval=$PARVALUELIST[$p] -v doesmatch=0 -f ${PARAMDOTAWK}`

	if ( "$nonMatchingLines" != "" ) then
	    if ($OUTPUT) then
		echo "lines number $nonMatchingLines will be commented out"
	    endif
	    foreach l ( $nonMatchingLines )
		echo "$l s/^ /\*/" >> $sedScript
	    end
	endif

	@ failed = 0

	if ( "$matchingLines" == "" ) then
	    if ( "$FORCE" == 0 ) then
		echo " ${thiscaller} : Failed setting $PARNAMELIST[$p] to $PARVALUELIST[$p] -- skipping"
		@ failed = 1
	    else
		# get the number of the first line defining the parameter
		set fline = `$GREP -n '$-PARAMETER' ${problemLoc} | $GREP $PARNAMELIST[$p] | $HEAD -1 | $AWK -F: '{print $1}'`
		echo "$fline s/^.*"'$'"/ IE $PARNAMELIST[$p]                   $PARVALUELIST[$p]/" >> $sedScript
	    endif
	else
	    if ($OUTPUT) then
		echo "$PARNAMELIST[$p] will be set to $PARVALUELIST[$p] on line $matchingLines"
	    endif
	    # change the leading star to a whitespace on the matching lines
	    # if there was no leading star, this has no effect.
            foreach ml ( $matchingLines )
	        echo "$ml s/^\*/ /" >> $sedScript
            end
	endif

	@ p++
    end

    # the sed script is ready; we now use it to cast the SIF file
    # note that the cast problem and the sed script have similar names
    if( $failed == 0 ) then
	$SED -f $sedScript $problemLoc > $sifFile
    else
	$LN -s $problemLoc $sifFile
    endif

endif

# If necessary, create a symbolic link between the current directory
# and the problem file
# Since the SIF decoder does not want file names longer than 10 chars,
# take the last 10 numbers of the process id, in an attempt to have a
# unique name.

if ( "$probDirNotGiven" == "true" && $nbparam == 0 ) then
    set TEMPNAME = $PROBNAME
    if ( $lookInMastSif == 0 ) then
	set link = "false"
    else
	set link = "true"
	$LN -s ${problemLoc} ./$TEMPNAME.SIF
    endif
else if ( $nbparam > 0 ) then
    set link = "true"
    set TEMPNAME = `echo $$ | $AWK 'BEGIN{nb=10}{l=length($1); if(l<nb){start=1}else{start=l-nb+1} print substr($1,start,nb)}'`
#    set TEMPNAME = "${PROBNAME}__${TEMPNAME}"
    $RM $TEMPNAME.SIF
    $LN -s $sifFile ./$TEMPNAME.SIF
else
    set link = "true"
    set TEMPNAME = `echo $$ | $AWK 'BEGIN{nb=10}{l=length($1); if(l<nb){start=1}else{start=l-nb+1} print substr($1,start,nb)}'`
#    set TEMPNAME = "${PROBNAME}__${TEMPNAME}"
    $RM $TEMPNAME.SIF
    $LN -s $PROBDIR/${PROBNAME}.SIF ./$TEMPNAME.SIF
endif

# Define the path to the decoder

set DECODER=$MYSIFDEC/double/bin/sifdec
if (! -e $DECODER ) then
  set DECODER=$MYSIFDEC/single/bin/sifdec
  if (! -e $DECODER ) then
    echo ' '
    echo 'No SIF decoder sifdec found in '$MYSIFDEC/double/bin
    echo 'or '$MYSIFDEC/single/bin
    echo 'Terminating execution.'
    exit 4
  endif
endif

if ($OUTPUT) then
  echo 'convert the sif file into data and routines suitable for optimizer...'
  echo ' '
  echo 'problem details will be given'
  echo ' '
endif

if (-e EXTERN.f) $RM EXTERN.f

# construct input file for the decoder

echo $TEMPNAME   > $TMP/sd${PAC}.input
echo $METHOD    >> $TMP/sd${PAC}.input
echo $OUTPUT    >> $TMP/sd${PAC}.input
echo $PROBNAME  >> $TMP/sd${PAC}.input
echo $automatic >> $TMP/sd${PAC}.input
echo $ad0       >> $TMP/sd${PAC}.input
echo $prec      >> $TMP/sd${PAC}.input

# Finally, decode the problem

$DECODER < $TMP/sd${PAC}.input

# Clean up

$RM $TMP/sd${PAC}.input
if ( $nbparam > 0 ) then
    $RM $sedScript
    $RM $sifFile
endif


if ( $link == "true" ) then
    $RM $TEMPNAME.SIF
endif

if (! -e OUTSDIF.d) then
  echo ' '
  echo "error exit from decoding stage. terminating execution."
  exit 3
endif

#  Rename files

if ( $automatic ) then
  if (-e ELFUND.f ) $MV ELFUND.f ELFUN.f
  if (-e GROUPD.f ) $MV GROUPD.f GROUP.f
  if (-e EXTERA.f ) $MV EXTERA.f EXTER.f
else
  if (-e ELFUNS.f ) $MV ELFUNS.f ELFUN.f
  if (-e GROUPS.f ) $MV GROUPS.f GROUP.f
  if (-e EXTERN.f ) $MV EXTERN.f EXTER.f
endif

if ( -e RANGES.f ) $MV RANGES.f RANGE.f

if (-e EXTER.f ) then
 if (-z EXTER.f ) $RM EXTER.f
endif

#  Record the type of derivatives used in the decoding

echo $automatic $ad0 > AUTOMAT.d

if ($OUTPUT) echo ' '

