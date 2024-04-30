#!/bin/bash
ssh seanzhou1207@arwen.berkeley.edu
dest="scp seanzhou1207@arwen.berkeley.edu:ps4.pdf ~/Documents/Git/stat243Git/seanzhou1207"

function my_scp() {
    if [ $# -ne 2 ]; then
        echo "Usage: my_scp <remote_file> <local_destination>"
        return 1
    fi
    
    remote_file="$1"
    local_destination="$2"
    
    scp seanzhou1207@arwen.berkeley.edu:"$remote_file" "$local_destination"
}



