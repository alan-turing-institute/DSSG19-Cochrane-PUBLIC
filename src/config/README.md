## Parameters for static and ignition .yaml files

### Static
`metrics`: evaluation metrics of interest

`classes`: review groups from our data which we consider for classification

### Ignition
`seed`: random seed so we can reproduce graphs

`table_name`: currently only pulling from papers but can pull from any in database

`ref_table`: for dict data types this calls the ref table

`unique_id`: id when joining from ref table

`results_table_name`: the psql table that we push results to

`target`: the variable in the table which are our labels

`existing_features`: list of existing variables in db

`generated_features`: list of features to be generated for this analysis

`model_type`: models which are to be called for this analysis

`query`: SQL query which generates the data that feeds this analysis

`train_neg_n`: The size of the testing set returned by the sample function

`test_perc`: the percent of positive labels saved for the test set

`features_table` : Features table (if one exists) from which features should be pulled. Note that in the provided yamls, features_table is often specified as `papers_features`, which is a relic from a former implementation - in the original implementation, all features are stored in a .pkl file.
