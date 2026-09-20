rm -f freq.summary   # safer: avoid error if file doesn't exist

for dir in $(ls -d */)   # only loop over directories
do
  if [ -d "$dir/freq" ]; then   # check if "freq" folder exists
    cd "$dir/freq"
    
    pwd >> ../../freq.summary
    echo -e "501\n298\n" | vaspkit >> ../../freq.summary
    
    cd ../../
  fi
done

grep '/freq' freq.summary
grep 'to G' freq.summary
