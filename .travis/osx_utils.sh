#!/bin/bash
# See: https://github.com/matthew-brett/multibuild/blob/devel/osx_utils.sh
LATEST_2p7=2.7.15
LATEST_2p6=2.6.6
LATEST_3p2=3.2.5
LATEST_3p3=3.3.5
LATEST_3p4=3.4.4
LATEST_3p5=3.5.4
LATEST_3p6=3.6.6
LATEST_3p7=3.7.0


function check_var {
    if [ -z "$1" ]; then
        echo "required variable not defined"
        exit 1
    fi
}


function fill_pyver {
    # Convert major or major.minor format to major.minor.micro
    #
    # Hence:
    # 2 -> 2.7.11  (depending on LATEST_2p7 value)
    # 2.7 -> 2.7.11  (depending on LATEST_2p7 value)
    local ver=$1
    check_var $ver
    if [[ $ver =~ [0-9]+\.[0-9]+\.[0-9]+ ]]; then
        # Major.minor.micro format already
        echo $ver
    elif [ $ver == 2 ] || [ $ver == "2.7" ]; then
        echo $LATEST_2p7
    elif [ $ver == "2.6" ]; then
        echo $LATEST_2p6
    elif [ $ver == 3 ] || [ $ver == "3.7" ]; then
        echo $LATEST_3p7
    elif [ $ver == "3.6" ]; then
        echo $LATEST_3p6
    elif [ $ver == "3.5" ]; then
        echo $LATEST_3p5
    elif [ $ver == "3.4" ]; then
        echo $LATEST_3p4
    elif [ $ver == "3.3" ]; then
        echo $LATEST_3p3
    elif [ $ver == "3.2" ]; then
        echo $LATEST_3p2
    else
        echo "Can't fill version $ver" 1>&2
        exit 1
    fi
}

