{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyPvxLJCUBnJ9nJVztHobZYC",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/yara-amr/CodeClause_Sales_Prediction/blob/main/CodeClause_Sales_Prediction.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 67,
      "metadata": {
        "id": "YAuJjE_KTa2H"
      },
      "outputs": [],
      "source": [
        "# Libraries & Packages:\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn import svm\n",
        "from sklearn.ensemble import RandomForestRegressor"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Data Preprocessing:\n",
        "sales_dataset = pd.read_csv('Train-Set.csv')\n",
        "sales_dataset.head()\n",
        "\n",
        "# Show NO of rows, col in loan_dataset\n",
        "sales_dataset.shape\n",
        "# Show statistical Measure\n",
        "sales_dataset.describe()\n",
        "\n",
        "# Show No of Missing Values in Each col\n",
        "sales_dataset.isnull().sum()\n",
        "# Dropping the missing values\n",
        "sales_dataset = sales_dataset.dropna()\n",
        "# No of Missing Values in Each col\n",
        "sales_dataset.isnull().sum()\n",
        "\n",
        "# Convert categorical col to numerical Val\n",
        "sales_dataset.replace({'FatContent' : {'Low Fat': 0, 'LF':0, 'low fat': 0, 'Regular': 1, 'reg':1}, 'OutletSize' : {'Small': 0, 'Medium': 1, 'High': 2},\n",
        "                       'LocationType': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}, \n",
        "                       'OutletType': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}}, inplace=True)\n",
        "\n",
        "sales_dataset.head()\n",
        "\n",
        "# Separating into Data, Label\n",
        "X_sales = sales_dataset.drop(columns= ['ProductID','ProductType', 'OutletID', 'OutletSales'], axis=1)\n",
        "Y_sales = sales_dataset['OutletSales'] \n",
        "print(X_sales)\n",
        "print(Y_sales)\n",
        "\n",
        "# Split Data into Training, and Test\n",
        "X_train, X_test, Y_train, Y_test = train_test_split(X_sales, Y_sales, test_size = 0.3, random_state = 0)\n",
        "print(X_sales.shape, X_train.shape, X_test.shape)"
      ],
      "metadata": {
        "id": "8FoGf1911WCo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Support Vector Regression Algorithm:\n",
        "\n",
        "sales_svr = svm.SVR(kernel = 'linear')\n",
        "sales_svr.fit(X_train, Y_train)\n",
        "\n",
        "y_pred = sales_svr.predict(X_test) \n",
        "\n",
        "# adding the list to the dataframe as column using assign(column_name = data)\n",
        "dataframe_svr = pd.DataFrame(X_test)\n",
        "sales_svr_df = dataframe_svr.assign(predictions = y_pred)\n",
        "\n",
        "sales_svr_df"
      ],
      "metadata": {
        "id": "UqJHF_XAiDc2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Random Forest Algorithm:\n",
        "\n",
        "sales_rf = RandomForestRegressor(random_state = 0)\n",
        "sales_rf.fit(X_train, Y_train)\n",
        "\n",
        "Y_pred = sales_rf.predict(X_test) \n",
        "\n",
        "# adding the list to the dataframe as column using assign(column_name = data)\n",
        "dataframe_rf = pd.DataFrame(X_test)\n",
        "sales_rf_df = dataframe_rf.assign(predictions = Y_pred)\n",
        "\n",
        "sales_rf_df"
      ],
      "metadata": {
        "id": "M0c_JwcNmXK5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Test Data Preprocessing:\n",
        "test_dataset = pd.read_csv('Test-Set.csv')\n",
        "test_dataset.head()\n",
        "\n",
        "# Show NO of rows, col in loan_dataset\n",
        "test_dataset.shape\n",
        "\n",
        "# Show statistical Measure\n",
        "test_dataset.describe()\n",
        "\n",
        "# Show No of Missing Values in Each col\n",
        "test_dataset.isnull().sum()\n",
        "# Dropping the missing values\n",
        "test_dataset = test_dataset.dropna()\n",
        "# No of Missing Values in Each col\n",
        "test_dataset.isnull().sum()\n",
        "\n",
        "# Convert categorical col to numerical Val\n",
        "test_dataset.replace({'FatContent' : {'Low Fat': 0, 'LF':0, 'low fat': 0, 'Regular': 1, 'reg':1}, 'OutletSize' : {'Small': 0, 'Medium': 1, 'High': 2},\n",
        "                       'LocationType': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}, \n",
        "                       'OutletType': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}}, inplace=True)\n",
        "\n",
        "test_dataset.head()\n",
        "test_dataset = test_dataset.drop(columns= ['ProductID','ProductType', 'OutletID'], axis=1)\n",
        "print(test_dataset)"
      ],
      "metadata": {
        "id": "hPfCDcpWhZTi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Prediction Using SVR\n",
        "pred_test_svr = sales_svr.predict(test_dataset)\n",
        "\n",
        "# Adding the list to the dataframe as column using assign(column_name = data)\n",
        "dataframe = pd.DataFrame(test_dataset)\n",
        "new_testdataset = dataframe.assign(predictions = pred_test_svr)\n",
        "\n",
        "print(\"\\nSVR Prediction: \") \n",
        "new_testdataset"
      ],
      "metadata": {
        "id": "ewpC1gX43GXx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Prediction Using RF\n",
        "pred_test_rf = sales_rf.predict(test_dataset)\n",
        "\n",
        "# Adding the list to the dataframe as column using assign(column_name = data)\n",
        "dataframe2 = pd.DataFrame(test_dataset)\n",
        "newtestdataset = dataframe2.assign(predictions = pred_test_rf)\n",
        "\n",
        "print(\"\\nRF Prediction: \") \n",
        "newtestdataset"
      ],
      "metadata": {
        "id": "yl2lta2v1-0p"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}