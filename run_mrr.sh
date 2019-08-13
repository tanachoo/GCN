
# Usage: sh run_mrr.sh

cd `dirname $0`

cv="0 1 2 3 4"
method=(ip2 distmult2 gcn2)

for m in ${method[@]}; do
	echo "method: $m"
	for i in $cv; do
    	echo "cv fold: $i"
		python mean_reciprocal_rank.py --input ./result/target2/score_${m}_cv${i}.txt --cv $i
	done
echo " "
done
