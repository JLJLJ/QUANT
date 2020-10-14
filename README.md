# QUANT
123

#!/bin/bash
read -p "Please input commit comment:" COMMENT
cd /home/jiang/ex/py/QUANT
git add .
git commit -m "$COMMENT"
git push -u origin master

