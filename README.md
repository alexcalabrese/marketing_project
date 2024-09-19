# Marketing_project

| Name           | Surname          | Ids   | Email                                   |
|----------------|------------------|-------|-----------------------------------------|
| Antonio        | Sabbatella        | 869432| a.sabbatella@campus.unimib.it           |
| Alex           | Calabrese         | 869054| a.calabrese19@campus.unimib.it          |
| Amelia Maria   | Acuna Rodriguez   | 910195| a.acunarodriguez@campus.unimib.it       |

## Important: replace the "base path" inside config.yaml file

Replace the "base path" inside  `configs.yaml` file with your own path to the data.

```yaml
data_paths:
  base_path: "<YOUR_PATH>/Marketing_data"
  customers: tbl_customers.csv
  ...
```
## Move to the `marketing_project` folder
Before running anything, make sure to be in the `marketing_project` folder root folder.

## IMPORTANT: create the `merged_data.csv` file
Run the `prepare_merged_data.py` file to create `merged_data.csv` file.

WARNING: all the notebooks and files are assuming this step, if not done, none of the files will work.

## Others

We didn't include in the repository the csv result of our analisys for Github's storage restriction policy.

To perform the Sentiment Analys the following python file is provided (it only needs to be run): `feedback_focus_sentiment.py`.
