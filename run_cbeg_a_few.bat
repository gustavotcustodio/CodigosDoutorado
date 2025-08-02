@echo off

echo "Clustering DBC external"
python cbeg.py -d german_credit -n compare -m 100.0 -p pso -e dbc_rand -c weighted_membership

echo "Clustering DBC"
python cbeg.py -d german_credit -n compare -m 100.0 -p pso -e dbc -c weighted_membership

echo "Clustering external"
python cbeg.py -d german_credit -n compare -m 100.0 -p pso -e rand -c weighted_membership

echo "Clustering DBC external"
python cbeg.py -d contraceptive -n compare -m 100.0 -p pso -e dbc_rand -c weighted_membership

echo "Clustering DBC"
python cbeg.py -d contraceptive -n compare -m 100.0 -p pso -e dbc -c weighted_membership

echo "Clustering external"
python cbeg.py -d contraceptive -n compare -m 100.0 -p pso -e rand -c weighted_membership
