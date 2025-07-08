@echo off

echo "Clustering DBC external"
python cbeg.py -d german_credit -n compare -m 100.0 -p clustering -e dbc_ext -c weighted_membership -b

echo "Clustering DBC"
python cbeg.py -d german_credit -n compare -m 100.0 -p clustering -e dbc -c weighted_membership -b

echo "Clustering external"
python cbeg.py -d german_credit -n compare -m 100.0 -p clustering -e ext -c weighted_membership -b

echo "Clustering DBC external"
python cbeg.py -d contraceptive -n compare -m 100.0 -p clustering -e dbc_ext -c weighted_membership -b

echo "Clustering DBC"
python cbeg.py -d contraceptive -n compare -m 100.0 -p clustering -e dbc -c weighted_membership -b

echo "Clustering external"
python cbeg.py -d contraceptive -n compare -m 100.0 -p clustering -e ext -c weighted_membership -b
