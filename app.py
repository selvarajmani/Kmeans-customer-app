{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 383,
   "id": "93e983b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 384,
   "id": "c39fbac5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Annual Income (k$)</th>\n",
       "      <th>Spending Score (1-100)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>78</td>\n",
       "      <td>male</td>\n",
       "      <td>58</td>\n",
       "      <td>133.0</td>\n",
       "      <td>70.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>34</td>\n",
       "      <td>Male</td>\n",
       "      <td>59</td>\n",
       "      <td>53.0</td>\n",
       "      <td>9.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>33</td>\n",
       "      <td>male</td>\n",
       "      <td>43</td>\n",
       "      <td>142.0</td>\n",
       "      <td>33.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>88</td>\n",
       "      <td>Male</td>\n",
       "      <td>59</td>\n",
       "      <td>111.0</td>\n",
       "      <td>68.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>80</td>\n",
       "      <td>male</td>\n",
       "      <td>56</td>\n",
       "      <td>52.0</td>\n",
       "      <td>54.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CustomerID Gender Age  Annual Income (k$)  Spending Score (1-100)\n",
       "0          78   male  58               133.0                    70.0\n",
       "1          34   Male  59                53.0                     9.0\n",
       "2          33   male  43               142.0                    33.0\n",
       "3          88   Male  59               111.0                    68.0\n",
       "4          80   male  56                52.0                    54.0"
      ]
     },
     "execution_count": 384,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv('Messy_Customer_Data.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 385,
   "id": "f2eecae8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 105 entries, 0 to 104\n",
      "Data columns (total 5 columns):\n",
      " #   Column                  Non-None Count  Dtype  \n",
      "---  ------                  --------------  -----  \n",
      " 0   CustomerID              105 non-None    int64  \n",
      " 1   Gender                  85 non-None     object \n",
      " 2   Age                     104 non-None    object \n",
      " 3   Annual Income (k$)      104 non-None    float64\n",
      " 4   Spending Score (1-100)  104 non-None    float64\n",
      "dtypes: float64(2), int64(1), object(2)\n",
      "memory usage: 4.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 386,
   "id": "740140a3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',\n",
       "       'Spending Score (1-100)'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 386,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 387,
   "id": "b7d68f9d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['male', 'Male', nan, 'Female', 'FEMALE'], dtype=object)"
      ]
     },
     "execution_count": 387,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 388,
   "id": "1b06f4b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Gender']=df['Gender'].replace({'male':'Male','FEMALE':'Female'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 389,
   "id": "ca3a5c5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Male', nan, 'Female'], dtype=object)"
      ]
     },
     "execution_count": 389,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 390,
   "id": "bc6f131b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Male      45\n",
       "Female    40\n",
       "Name: Gender, dtype: int64"
      ]
     },
     "execution_count": 390,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 391,
   "id": "cd64dcf3",
   "metadata": {},
   "outputs": [],
   "source": [
    "mod_gen=df['Gender'].mode()[0]\n",
    "df['Gender']=df['Gender'].fillna(value=mod_gen)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 392,
   "id": "5221655b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Male', 'Female'], dtype=object)"
      ]
     },
     "execution_count": 392,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 393,
   "id": "9d590c7c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Male      65\n",
       "Female    40\n",
       "Name: Gender, dtype: int64"
      ]
     },
     "execution_count": 393,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 394,
   "id": "9a986bd6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now lets check the Age column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 395,
   "id": "72ed3fd6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['58', '59', '43', '56', '32', '54', '69', '53', '62', '39', '28',\n",
       "       '26', '24', '46', '44', '25', '61', 'twenty', '31', '45', '30',\n",
       "       '65', '42', '51', '18', '49', '20', '41', '47', '29', '57', '21',\n",
       "       '40', '66', '37', '23', '22', nan, '52', '50', '68', '60', '64',\n",
       "       '19'], dtype=object)"
      ]
     },
     "execution_count": 395,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Age'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 396,
   "id": "d9f0431a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['58', '59', '43', '56', '32', '54', '69', '53', '62', '39', '28',\n",
       "       '26', '24', '46', '44', '25', '61', '20', '31', '45', '30', '65',\n",
       "       '42', '51', '18', '49', '41', '47', '29', '57', '21', '40', '66',\n",
       "       '37', '23', '22', nan, '52', '50', '68', '60', '64', '19'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 396,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Age']=df['Age'].replace({'twenty':'20'})\n",
    "df['Age'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 397,
   "id": "ecbb7573",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<IntegerArray>\n",
       "[  58,   59,   43,   56,   32,   54,   69,   53,   62,   39,   28,   26,   24,\n",
       "   46,   44,   25,   61,   20,   31,   45,   30,   65,   42,   51,   18,   49,\n",
       "   41,   47,   29,   57,   21,   40,   66,   37,   23,   22, <NA>,   52,   50,\n",
       "   68,   60,   64,   19]\n",
       "Length: 43, dtype: Int64"
      ]
     },
     "execution_count": 397,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Age']=df['Age'].astype('Int64')\n",
    "df['Age'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 398,
   "id": "b8c5ad50",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([133.,  53., 142., 111.,  52.,  62.,  74., 113., 143., 138., 140.,\n",
       "        47., 144.,  68.,  23.,  22., 145., 124.,  41.,  83.,  26., 129.,\n",
       "        95., 115.,  66.,  16., 149.,  84.,  21.,  18.,  25.,  36.,  63.,\n",
       "        35.,  96.,  82., 126., 107., 127., 110., 130., 104.,  98.,  56.,\n",
       "        89., 123., 125.,  60.,  49.,  75., 137., 118., 131.,  67.,  44.,\n",
       "       101., 134.,  65.,  31.,  51., 112.,  nan,  20.,  38.,  42.,  17.])"
      ]
     },
     "execution_count": 398,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Annual Income (k$)'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 399,
   "id": "910cb5a8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      133\n",
       "1       53\n",
       "2      142\n",
       "3      111\n",
       "4       52\n",
       "      ... \n",
       "100    142\n",
       "101    127\n",
       "102     82\n",
       "103    127\n",
       "104    115\n",
       "Name: Annual Income (k$), Length: 105, dtype: Int64"
      ]
     },
     "execution_count": 399,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mod_ai=df['Annual Income (k$)'].mean()\n",
    "df['Annual Income (k$)']=df['Annual Income (k$)'].fillna(value=mod_ai).round()\n",
    "df['Annual Income (k$)'].unique()\n",
    "\n",
    "df['Annual Income (k$)'].astype('Int64')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 400,
   "id": "7abf1c3b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🔁 Full Row Duplicates:\n",
      "      CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)\n",
      "35           77    Male   30               129.0                    42.0\n",
      "63           44  Female   54                74.0                    24.0\n",
      "70           18    Male   18                82.0                    95.0\n",
      "87           93  Female   49                18.0                    99.0\n",
      "104           6  Female   51               115.0                    95.0\n"
     ]
    }
   ],
   "source": [
    "# Find duplicate rows (excluding the first occurrence)\n",
    "duplicate_rows = df[df.duplicated()]\n",
    "\n",
    "# Show them\n",
    "print(\"🔁 Full Row Duplicates:\\n\", duplicate_rows)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 401,
   "id": "2c4a03d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)\n",
       "77          Male    30   129.0               42.0                      2\n",
       "6           Female  51   115.0               95.0                      2\n",
       "44          Female  54   74.0                24.0                      2\n",
       "93          Female  49   18.0                99.0                      2\n",
       "18          Male    18   82.0                95.0                      2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 401,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.value_counts().loc[lambda x: x > 1]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 402,
   "id": "6771fc61",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop_duplicates(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 403,
   "id": "4c457c24",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.rename(columns={'Annual Income (k$)': 'Annual_income', \n",
    "                        'Spending Score (1-100)': 'Spending_score'})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 404,
   "id": "a582c299",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 100 entries, 0 to 103\n",
      "Data columns (total 5 columns):\n",
      " #   Column          Non-Null Count  Dtype  \n",
      "---  ------          --------------  -----  \n",
      " 0   CustomerID      100 non-None    int64  \n",
      " 1   Gender          100 non-None    object \n",
      " 2   Age             99 non-None     Int64  \n",
      " 3   Annual_income   100 non-None    float64\n",
      " 4   Spending_score  99 non-None     float64\n",
      "dtypes: Int64(1), float64(2), int64(1), object(1)\n",
      "memory usage: 4.8+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 405,
   "id": "a0ef414f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['CustomerID', 'Gender', 'Age', 'Annual_income', 'Spending_score'], dtype='object')"
      ]
     },
     "execution_count": 405,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 406,
   "id": "c8c5182e",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Annual_income']=df['Annual_income'].astype('Int64').round()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 407,
   "id": "c9cf0225",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Spending_score']=df['Spending_score'].astype('Int64').round()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 408,
   "id": "f3f67890",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Annual_income</th>\n",
       "      <th>Spending_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>78</td>\n",
       "      <td>Male</td>\n",
       "      <td>58</td>\n",
       "      <td>133</td>\n",
       "      <td>70</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>34</td>\n",
       "      <td>Male</td>\n",
       "      <td>59</td>\n",
       "      <td>53</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>33</td>\n",
       "      <td>Male</td>\n",
       "      <td>43</td>\n",
       "      <td>142</td>\n",
       "      <td>33</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>88</td>\n",
       "      <td>Male</td>\n",
       "      <td>59</td>\n",
       "      <td>111</td>\n",
       "      <td>68</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>80</td>\n",
       "      <td>Male</td>\n",
       "      <td>56</td>\n",
       "      <td>52</td>\n",
       "      <td>54</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CustomerID Gender  Age  Annual_income  Spending_score\n",
       "0          78   Male   58            133              70\n",
       "1          34   Male   59             53               9\n",
       "2          33   Male   43            142              33\n",
       "3          88   Male   59            111              68\n",
       "4          80   Male   56             52              54"
      ]
     },
     "execution_count": 408,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 409,
   "id": "83759d0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# customers who spend above 100k and spending score is above 80"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 410,
   "id": "3537006e",
   "metadata": {},
   "outputs": [],
   "source": [
    "val_cus=df[(df['Annual_income']>100) & (df['Spending_score']>80)]['CustomerID']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 411,
   "id": "a774eb0a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     26\n",
       "1      3\n",
       "2     65\n",
       "3      6\n",
       "4     74\n",
       "5     11\n",
       "6     63\n",
       "7     31\n",
       "8     90\n",
       "9      9\n",
       "10    59\n",
       "11    13\n",
       "12     7\n",
       "Name: CustomerID, dtype: int64"
      ]
     },
     "execution_count": 411,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "val_cus.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 412,
   "id": "fed649a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Age              42.793651\n",
      "Annual_income    82.365079\n",
      "dtype: float64\n",
      "Age               48.944444\n",
      "Annual_income    100.891892\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "male_avg_age=df[(df['Gender']=='Male')][['Age','Annual_income']].mean()\n",
    "female_avg_age=df[(df['Gender']=='Female')][['Age','Annual_income']].mean()\n",
    "\n",
    "print(male_avg_age)\n",
    "print(female_avg_age)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 413,
   "id": "18bda5f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Replace string garbage values if missed\n",
    "df['Annual_income'] = df['Annual_income'].replace('fifty', np.nan).astype('float')\n",
    "df['Spending_score'] = df['Spending_score'].replace('N/A', np.nan).astype('float')\n",
    "\n",
    "# Step 2: Fill missing values (you can use mean or drop them)\n",
    "df[['Annual_income', 'Spending_score']] = df[['Annual_income', 'Spending_score']].fillna(df[['Annual_income', 'Spending_score']].mean())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 414,
   "id": "556be94a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def classify_customer(row):\n",
    "    if row['Annual_income'] > 100 and row['Spending_score'] > 80:\n",
    "        return 'High value'\n",
    "    elif row['Annual_income'] > 60 and row['Spending_score'] > 50:\n",
    "        return 'Medium value'\n",
    "    else:\n",
    "        return 'Low value'\n",
    "\n",
    "df['Customer Type'] = df.apply(classify_customer, axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 415,
   "id": "0ce5f28b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Annual_income</th>\n",
       "      <th>Spending_score</th>\n",
       "      <th>Customer Type</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>78</td>\n",
       "      <td>Male</td>\n",
       "      <td>58</td>\n",
       "      <td>133.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>Medium value</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>34</td>\n",
       "      <td>Male</td>\n",
       "      <td>59</td>\n",
       "      <td>53.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>Low value</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>33</td>\n",
       "      <td>Male</td>\n",
       "      <td>43</td>\n",
       "      <td>142.0</td>\n",
       "      <td>33.0</td>\n",
       "      <td>Low value</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>88</td>\n",
       "      <td>Male</td>\n",
       "      <td>59</td>\n",
       "      <td>111.0</td>\n",
       "      <td>68.0</td>\n",
       "      <td>Medium value</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>80</td>\n",
       "      <td>Male</td>\n",
       "      <td>56</td>\n",
       "      <td>52.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>Low value</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CustomerID Gender  Age  Annual_income  Spending_score Customer Type\n",
       "0          78   Male   58          133.0            70.0  Medium value\n",
       "1          34   Male   59           53.0             9.0     Low value\n",
       "2          33   Male   43          142.0            33.0     Low value\n",
       "3          88   Male   59          111.0            68.0  Medium value\n",
       "4          80   Male   56           52.0            54.0     Low value"
      ]
     },
     "execution_count": 415,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 416,
   "id": "e7146b9b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CustomerID          int64\n",
       "Gender             object\n",
       "Age                 Int64\n",
       "Annual_income     float64\n",
       "Spending_score    float64\n",
       "Customer Type      object\n",
       "dtype: object"
      ]
     },
     "execution_count": 416,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 417,
   "id": "6d2ab9aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preprocessing for kmeans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 418,
   "id": "043ae63d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Annual_income</th>\n",
       "      <th>Spending_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>133.0</td>\n",
       "      <td>70.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>53.0</td>\n",
       "      <td>9.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>142.0</td>\n",
       "      <td>33.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>111.0</td>\n",
       "      <td>68.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>52.0</td>\n",
       "      <td>54.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99</th>\n",
       "      <td>17.0</td>\n",
       "      <td>27.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100</th>\n",
       "      <td>142.0</td>\n",
       "      <td>49.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>101</th>\n",
       "      <td>127.0</td>\n",
       "      <td>15.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102</th>\n",
       "      <td>82.0</td>\n",
       "      <td>58.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>103</th>\n",
       "      <td>127.0</td>\n",
       "      <td>99.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>100 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Annual_income  Spending_score\n",
       "0            133.0            70.0\n",
       "1             53.0             9.0\n",
       "2            142.0            33.0\n",
       "3            111.0            68.0\n",
       "4             52.0            54.0\n",
       "..             ...             ...\n",
       "99            17.0            27.0\n",
       "100          142.0            49.0\n",
       "101          127.0            15.0\n",
       "102           82.0            58.0\n",
       "103          127.0            99.0\n",
       "\n",
       "[100 rows x 2 columns]"
      ]
     },
     "execution_count": 418,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df[['Annual_income', 'Spending_score']]\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 419,
   "id": "202911a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 420,
   "id": "955a3026",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 421,
   "id": "0ecc739f",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler=StandardScaler()\n",
    "x_scaled=scaler.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 422,
   "id": "a201ec1a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.06134297,  0.69693598],\n",
       "       [-0.87806858, -1.48945175],\n",
       "       [ 1.27952677, -0.62923363],\n",
       "       [ 0.52800479,  0.62525114],\n",
       "       [-0.90231122,  0.12345723],\n",
       "       [-0.65988478,  0.2309845 ],\n",
       "       [-0.36897305, -0.95181542],\n",
       "       [ 0.57649008, -0.88013058],\n",
       "       [ 0.57649008, -1.09518511],\n",
       "       [ 1.30376941, -1.56113659],\n",
       "       [ 1.18255619,  1.27041473],\n",
       "       [ 1.23104148,  0.58940871],\n",
       "       [-1.02352445,  0.44603903],\n",
       "       [ 1.32801206,  1.52131168],\n",
       "       [-0.51442891, -1.2385548 ],\n",
       "       [-1.60534791, -0.19912457],\n",
       "       [-1.62959055,  1.66468137],\n",
       "       [ 1.3522547 , -0.16328214],\n",
       "       [ 0.84315917,  1.27041473],\n",
       "       [-1.16898031,  0.08761481],\n",
       "       [-0.65988478, -0.12743972],\n",
       "       [-0.90231122,  0.26682692],\n",
       "       [-0.15078925, -0.16328214],\n",
       "       [-1.53261998, -0.48586394],\n",
       "       [ 1.27952677,  0.26682692],\n",
       "       [ 0.96437239, -0.30665183],\n",
       "       [ 0.14012248,  0.33851176],\n",
       "       [-0.65988478, -1.02350027],\n",
       "       [ 0.62497537,  1.59299652],\n",
       "       [-0.5629142 , -1.20271238],\n",
       "       [-1.77504642,  0.33851176],\n",
       "       [ 1.44922528,  1.27041473],\n",
       "       [-0.87806858, -0.95181542],\n",
       "       [-0.1265466 , -0.98765785],\n",
       "       [-1.6538332 ,  1.73636621],\n",
       "       [-1.72656113,  1.73636621],\n",
       "       [-1.55686262,  1.52131168],\n",
       "       [-1.29019353, -1.27439722],\n",
       "       [-0.63564214, -0.37833667],\n",
       "       [-1.31443618,  0.41019661],\n",
       "       [ 0.16436513,  0.01592997],\n",
       "       [-0.17503189,  1.59299652],\n",
       "       [ 1.32801206,  1.23457231],\n",
       "       [ 0.89164446, -0.915973  ],\n",
       "       [ 0.57649008, -1.13102753],\n",
       "       [ 0.43103422,  1.30625715],\n",
       "       [ 0.16436513, -0.27080941],\n",
       "       [ 0.9158871 ,  0.08761481],\n",
       "       [ 0.50376215,  1.41378442],\n",
       "       [ 1.30376941, -0.62923363],\n",
       "       [ 0.98861504,  0.91199051],\n",
       "       [ 0.35830628, -0.01991246],\n",
       "       [ 0.21285042,  1.70052379],\n",
       "       [ 0.9158871 , -1.34608206],\n",
       "       [-0.80534065, -0.30665183],\n",
       "       [-0.00533338,  1.30625715],\n",
       "       [-0.51442891, -0.915973  ],\n",
       "       [ 0.81891653,  0.33851176],\n",
       "       [ 1.3522547 , -1.52529417],\n",
       "       [ 0.86740182,  1.62883895],\n",
       "       [-0.70837007, -1.2385548 ],\n",
       "       [-1.02352445,  0.05177239],\n",
       "       [ 0.52800479, -0.12743972],\n",
       "       [-0.97503916,  0.15929965],\n",
       "       [ 1.32801206, -1.16686995],\n",
       "       [-0.65988478,  0.51772387],\n",
       "       [ 0.43103422,  0.55356629],\n",
       "       [ 0.9158871 ,  1.66468137],\n",
       "       [ 0.89164446,  0.55356629],\n",
       "       [-1.16898031, -1.52529417],\n",
       "       [-0.3447304 ,  1.5571541 ],\n",
       "       [ 1.15831355,  0.12345723],\n",
       "       [ 0.6977033 ,  1.48546926],\n",
       "       [ 1.01285768, -1.34608206],\n",
       "       [-1.31443618,  0.33851176],\n",
       "       [-0.53867156, -0.915973  ],\n",
       "       [ 0.35830628, -1.7403487 ],\n",
       "       [-1.09625238, -1.63282144],\n",
       "       [ 0.28557835,  1.30625715],\n",
       "       [ 1.08558561, -1.31023964],\n",
       "       [-0.51442891, -0.66507605],\n",
       "       [ 1.30376941, -0.4141791 ],\n",
       "       [ 0.57649008, -0.66507605],\n",
       "       [ 0.98861504, -0.95181542],\n",
       "       [-0.58715685,  0.        ],\n",
       "       [-1.41140675,  0.05177239],\n",
       "       [-0.92655387, -0.52170636],\n",
       "       [ 1.27952677,  0.44603903],\n",
       "       [ 0.55224744, -0.66507605],\n",
       "       [-0.00533338,  0.26682692],\n",
       "       [-1.77504642,  0.44603903],\n",
       "       [-1.67807584, -0.73676089],\n",
       "       [-1.24170824,  0.58940871],\n",
       "       [-1.14473767, -1.59697902],\n",
       "       [ 0.6977033 , -0.0915973 ],\n",
       "       [-1.75080378, -0.84428816],\n",
       "       [ 1.27952677, -0.05575488],\n",
       "       [ 0.9158871 , -1.27439722],\n",
       "       [-0.17503189,  0.26682692],\n",
       "       [ 0.9158871 ,  1.73636621]])"
      ]
     },
     "execution_count": 422,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 425,
   "id": "f3996427",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.cluster import KMeans\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "wcss = []\n",
    "for k in range(1, 11):\n",
    "    kmeans = KMeans(n_clusters=k, random_state=42)\n",
    "    kmeans.fit(x_scaled)\n",
    "    wcss.append(kmeans.inertia_)\n",
    "\n",
    "plt.plot(range(1, 11), wcss, marker='o')\n",
    "plt.xlabel('Number of Clusters (k)')\n",
    "plt.ylabel('WCSS')\n",
    "plt.title('Elbow Method')\n",
    "plt.grid(True)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 427,
   "id": "411531be",
   "metadata": {},
   "outputs": [],
   "source": [
    "kmeans = KMeans(n_clusters=5, random_state=42)\n",
    "df['Cluster'] = kmeans.fit_predict(x_scaled)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 428,
   "id": "687d8b96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjwAAAHFCAYAAAD2eiPWAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8qNh9FAAAACXBIWXMAAA9hAAAPYQGoP6dpAACX/ElEQVR4nOzdd3iT5frA8W/2aro3lFL2HrIElCEC4sS9tx6OoB5Ej8rhqKgIioqouAfy04N7KyC4QERlD5myR1u6d5v5/v6oFEIToG2apOn9ua5e2ud5k/fO09DeeaZKURQFIYQQQogwpg52AEIIIYQQjU0SHiGEEEKEPUl4hBBCCBH2JOERQgghRNiThEcIIYQQYU8SHiGEEEKEPUl4hBBCCBH2JOERQgghRNiThEcIIYQQYU8SHiFC2MaNG7n55pvJyMjAaDQSERHBaaedxsyZMykoKGiUey5YsICpU6c2ynOHggMHDjB+/Hg6dOiAyWQiNjaW7t27c/vtt3PgwIFghxdwL7/8Mu+8806wwxCi0ankaAkhQtMbb7zB+PHj6dixI+PHj6dLly44HA5Wr17NG2+8Qc+ePfn888/9ft8777yTl156iXD81XDw4EF69+5NdHQ09957Lx07dqS4uJgtW7bw0Ucf8fzzzzN06NBghxlQ3bp1Iz4+np9//jnYoQjRqLTBDkAIUdtvv/3GHXfcwciRI/niiy8wGAw1dSNHjuTee+9l0aJFQYwwdFVWVmI0GlGpVLXq3njjDfLy8li5ciUZGRk15WPHjuU///kPbrc7kKEKIQJIhrSECEHTp09HpVLx+uuveyQ7R+j1ei688MKa71UqlddhqNatW3PTTTfVfF9RUcF9991XM0QWGxtL3759ef/99wG46aabeOmll2qe88jX3r17AaiqqmLy5MlkZGSg1+tp0aIFEyZMoKioqNZ9zz//fL755ht69+6NyWSic+fOfPPNNwC88847dO7cGYvFQv/+/Vm9enWt2FevXs2FF15IbGwsRqOR3r1789FHH3lc884776BSqVi8eDG33HILCQkJmM1mbDab13bNz89HrVaTmJjotV6t9vyVeCoxACxfvpyBAwdiNBpp0aIFDz30EG+++aZH2wWjXX766SfuuOMO4uPjiYuL45JLLiEzM9Mjns2bN7N06dKan3Xr1q0BcLvdTJs2jY4dO2IymYiOjqZHjx48//zzXttOiJCnCCFCitPpVMxmszJgwIBTfgygPPLII7XK09PTlRtvvLHm+3Hjxilms1mZNWuW8tNPPynffPON8uSTTyovvviioiiKsnPnTuWyyy5TAOW3336r+aqqqlLcbrcyevRoRavVKg899JCyePFi5ZlnnlEsFovSu3dvpaqqyuO+LVu2VLp166a8//77yoIFC5QBAwYoOp1Oefjhh5XBgwcrn332mfL5558rHTp0UJKSkpSKioqax//444+KXq9XzjzzTOXDDz9UFi1apNx0000KoMydO7fmurlz5yqA0qJFC+Uf//iHsnDhQuWTTz5RnE6n13Z67733FEAZNWqUsmjRIqW4uNhnm55qDBs2bFCMRqPSo0cP5YMPPlC++uor5dxzz1Vat26tAMqePXuC1i5t2rRR7rrrLuW7775T3nzzTSUmJkYZPnx4zXVr165V2rRpo/Tu3bvmZ7127VpFURRlxowZikajUR555BHlhx9+UBYtWqTMnj1bmTp1qs82EyKUScIjRIjJzs5WAOWqq6465cecasLTrVs3ZezYsSd8rgkTJijePgstWrRIAZSZM2d6lH/44YcKoLz++use9zWZTMrBgwdrytavX68ASkpKilJeXl5T/sUXXyiA8tVXX9WUderUSendu7ficDg87nX++ecrKSkpisvlUhTl6B/2G2644YSv6Qi3262MGzdOUavVCqCoVCqlc+fOyj333OORmNQlhssvv1yxWCxKbm5uzTUul0vp0qWL14QnkO0yfvx4j+tmzpypAEpWVlZNWdeuXZWhQ4fWaqvzzz9f6dWrl7dmFKJJkiEtIZqR/v37s3DhQh588EF+/vlnKisrT/mxP/74I4DHEBnA5ZdfjsVi4YcffvAo79WrFy1atKj5vnPnzgAMGzYMs9lcq3zfvn0A7Ny5k23btnHttdcC4HQ6a77OPfdcsrKy2L59u8e9Lr300lN6DSqVildffZXdu3fz8ssvc/PNN+NwOHjuuefo2rUrS5curXMMS5cu5ayzziI+Pr7mPmq1miuuuMJrDIFsl2OHPQF69Ojh8Zwn0r9/fzZs2MD48eP57rvvKCkpOeljhAhlkvAIEWLi4+Mxm83s2bPH78/9wgsv8MADD/DFF18wfPhwYmNjGTt2LH/99ddJH5ufn49WqyUhIcGjXKVSkZycTH5+vkd5bGysx/d6vf6E5VVVVQAcPnwYgPvuuw+dTufxNX78eADy8vI8niMlJeWk8R8rPT2dO+64g7feeou//vqLDz/8kKqqKv7973/XOYb8/HySkpJq3cNb2Ylef2O0S1xcnMf3R+aDnUqiO3nyZJ555hl+//13xowZQ1xcHCNGjPA6r0iIpkBWaQkRYjQaDSNGjGDhwoUcPHiQli1bnvQxBoPB60Td45MQi8XCo48+yqOPPsrhw4drensuuOACtm3bdsJ7xMXF4XQ6yc3N9Uh6FEUhOzubfv36neIrPLEjPSWTJ0/mkksu8XpNx44dPb73tiKrLq644gpmzJjBn3/+WecY4uLiapKRY2VnZzcopuPVp10aQqvVMmnSJCZNmkRRURHff/89//nPfxg9ejQHDhzw6I0SoimQHh4hQtDkyZNRFIXbb78du91eq97hcPD111/XfN+6dWs2btzocc2PP/5IWVmZz3skJSVx0003cfXVV7N9+3YqKioA370AI0aMAOC9997zKP/0008pLy+vqW+ojh070r59ezZs2EDfvn29flmt1no9d1ZWltfysrIyDhw4QGpqap1jGDp0KD/++KNH74rb7ebjjz+uV4y+NFa7GAyGk/b4REdHc9lllzFhwgQKCgo8Vp4J0VRID48QIWjgwIG88sorjB8/nj59+nDHHXfQtWtXHA4H69at4/XXX6dbt25ccMEFAFx//fU89NBDPPzwwwwdOpQtW7YwZ84coqKiPJ53wIABnH/++fTo0YOYmBi2bt3Ku+++y8CBA2s+sXfv3h2Ap556ijFjxqDRaOjRowcjR45k9OjRPPDAA5SUlDB48GA2btzII488Qu/evbn++uv99vpfe+01xowZw+jRo7npppto0aIFBQUFbN26lbVr19Y7mXjiiSf49ddfufLKK+nVqxcmk4k9e/YwZ84c8vPzefrpp+scw5QpU/j6668ZMWIEU6ZMwWQy8eqrr1JeXg7UXureEI3RLt27d+eDDz7gww8/pE2bNhiNRrp3784FF1xAt27d6Nu3LwkJCezbt4/Zs2eTnp5O+/bt/faahAiYYM+aFkL4tn79euXGG29UWrVqpej1+pol4A8//LCSk5NTc53NZlPuv/9+JS0tTTGZTMrQoUOV9evX11ql9eCDDyp9+/ZVYmJiFIPBoLRp00a55557lLy8PI/nuu2225SE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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "sns.scatterplot(x='Annual_income', y='Spending_score', hue='Cluster', data=df, palette='Set1')\n",
    "plt.title('Customer Segments')\n",
    "plt.grid(True)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 429,
   "id": "c3086a57",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Annual_income</th>\n",
       "      <th>Spending_score</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cluster</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>39.00</td>\n",
       "      <td>64.36</td>\n",
       "      <td>46.43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>117.18</td>\n",
       "      <td>55.59</td>\n",
       "      <td>43.38</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>116.83</td>\n",
       "      <td>90.50</td>\n",
       "      <td>43.17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>53.29</td>\n",
       "      <td>22.81</td>\n",
       "      <td>46.33</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>127.94</td>\n",
       "      <td>20.72</td>\n",
       "      <td>45.67</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Annual_income  Spending_score    Age\n",
       "Cluster                                      \n",
       "0                39.00           64.36  46.43\n",
       "1               117.18           55.59  43.38\n",
       "2               116.83           90.50  43.17\n",
       "3                53.29           22.81  46.33\n",
       "4               127.94           20.72  45.67"
      ]
     },
     "execution_count": 429,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby('Cluster')[['Annual_income', 'Spending_score', 'Age']].mean().round(2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 430,
   "id": "1728b7a8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    22\n",
       "3    21\n",
       "0    21\n",
       "4    18\n",
       "2    18\n",
       "Name: Cluster, dtype: int64"
      ]
     },
     "execution_count": 430,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Cluster'].value_counts()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 431,
   "id": "fb9fdbe0",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "35",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3628\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3629\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3630\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.Int64HashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.Int64HashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 35",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_11244\\3264071819.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mscatterplot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Annual_income'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Spending_score'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mhue\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Cluster'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mpalette\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'Set2'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m     \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtext\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Annual_income'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdf\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Spending_score'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdf\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'Cluster'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfontsize\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m6\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtitle\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Customer Segmentation with Cluster Labels'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgrid\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\series.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    956\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    957\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0mkey_is_scalar\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 958\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_value\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    959\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    960\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mis_hashable\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\series.py\u001b[0m in \u001b[0;36m_get_value\u001b[1;34m(self, label, takeable)\u001b[0m\n\u001b[0;32m   1067\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1068\u001b[0m         \u001b[1;31m# Similar to Index.get_value, but we do not fall back to positional\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1069\u001b[1;33m         \u001b[0mloc\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlabel\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1070\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_values_for_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mloc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1071\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3629\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3630\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3631\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3632\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3633\u001b[0m                 \u001b[1;31m# If we have a listlike key, _check_indexing_error will raise\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 35"
     ]
    },
    {
     "data": {
      "image/png": "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fshZWilA0zTuuOMOTj31VAYMGABAZWUlAAUFBRHXFhQUUFFRkfAYE+GNN95gyZIlLF68uFlfR7sfmzdv5vnnn+eOO+7g//7v/1i4cCG33norJpOJa665psPdj3vuuYfa2lr69u2LTqcjGAzyyCOP8NOf/hToeD8fhzqa115ZWYnRaCQrK6vZNQe+Px15PB7uvfderrjiivDZUR3tXjz22GPo9XpuvfXWqP0d6X5IwpMCbr75ZlasWMG8efOa9R1+erumaSlxonusbdu2jdtuu42ZM2diNptbvK6j3I9QKMTw4cN59NFHARg6dCirV6/m+eef55prrglf11Hux5tvvsmrr77KjBkz6N+/P8uXL2fKlCkUFxdz7bXXhq/rKPcjmra89nS+P36/n8svv5xQKMRzzz13xOvT8V4sWbKEP//5zyxduvSYX1s63g+Z0kqyW265hQ8++IAvv/wy4sT2wsJCgGYZdlVVVbPf5NLBkiVLqKqqYtiwYej1evR6PbNnz+Yvf/kLer0+/Jo7yv0oKiqirKwsoq1fv35s3boV6Hg/H7/+9a+59957ufzyyxk4cCBXX301t99+O9OmTQM63v041NG89sLCQnw+H9XV1S1ek078fj8/+clPKC8vZ9asWREng3ekezF37lyqqqro0qVL+H21oqKCO++8k27dugEd635IwpMkmqZx88038+677/LFF19QWloa0V9aWkphYSGzZs0Kt/l8PmbPns3IkSMTHW7cnXXWWaxcuZLly5eHv4YPH86VV17J8uXL6d69e4e6H6NGjWpWpmD9+vV07doV6Hg/H42Njahq5NuVTqcLb0vvaPfjUEfz2ocNG4bBYIi4ZteuXaxatSrt7s+BZGfDhg189tln5OTkRPR3pHtx9dVXs2LFioj31eLiYn7961/z6aefAh3rfsgurST51a9+pTmdTu2rr77Sdu3aFf5qbGwMX/OHP/xBczqd2rvvvqutXLlS++lPf6oVFRVpdXV1SYw8cQ7dpaVpHet+LFy4UNPr9dojjzyibdiwQXvttdc0q9Wqvfrqq+FrOtL9uPbaa7VOnTppH330kVZeXq69++67Wm5urnb33XeHr0nn+1FfX68tW7ZMW7ZsmQZoTz75pLZs2bLwzqOjee033nij1rlzZ+2zzz7Tli5dqp155pna4MGDtUAgkKyX1Sat3Qu/369d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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x='Annual_income', y='Spending_score', hue='Cluster', data=df, palette='Set2')\n",
    "for i in range(df.shape[0]):\n",
    "    plt.text(df['Annual_income'][i], df['Spending_score'][i], str(df['Cluster'][i]), fontsize=6)\n",
    "plt.title('Customer Segmentation with Cluster Labels')\n",
    "plt.grid(True)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 432,
   "id": "ff4f3481",
   "metadata": {},
   "outputs": [],
   "source": [
    "def label_cluster(cluster):\n",
    "    if cluster == 0:\n",
    "        return 'Target'\n",
    "    elif cluster == 1:\n",
    "        return 'Careful Spender'\n",
    "    elif cluster == 2:\n",
    "        return 'Luxury Seeker'\n",
    "    # ... add for all clusters\n",
    "    else:\n",
    "        return 'Others'\n",
    "\n",
    "df['Cluster_Label'] = df['Cluster'].apply(label_cluster)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 433,
   "id": "9e757aba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Annual_income</th>\n",
       "      <th>Spending_score</th>\n",
       "      <th>Customer Type</th>\n",
       "      <th>Cluster</th>\n",
       "      <th>Cluster_Label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>78</td>\n",
       "      <td>Male</td>\n",
       "      <td>58</td>\n",
       "      <td>133.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>Medium value</td>\n",
       "      <td>1</td>\n",
       "      <td>Careful Spender</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>34</td>\n",
       "      <td>Male</td>\n",
       "      <td>59</td>\n",
       "      <td>53.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>Low value</td>\n",
       "      <td>3</td>\n",
       "      <td>Others</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>33</td>\n",
       "      <td>Male</td>\n",
       "      <td>43</td>\n",
       "      <td>142.0</td>\n",
       "      <td>33.0</td>\n",
       "      <td>Low value</td>\n",
       "      <td>4</td>\n",
       "      <td>Others</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>88</td>\n",
       "      <td>Male</td>\n",
       "      <td>59</td>\n",
       "      <td>111.0</td>\n",
       "      <td>68.0</td>\n",
       "      <td>Medium value</td>\n",
       "      <td>1</td>\n",
       "      <td>Careful Spender</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>80</td>\n",
       "      <td>Male</td>\n",
       "      <td>56</td>\n",
       "      <td>52.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>Low value</td>\n",
       "      <td>0</td>\n",
       "      <td>Target</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99</th>\n",
       "      <td>29</td>\n",
       "      <td>Male</td>\n",
       "      <td>44</td>\n",
       "      <td>17.0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>Low value</td>\n",
       "      <td>3</td>\n",
       "      <td>Others</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100</th>\n",
       "      <td>75</td>\n",
       "      <td>Male</td>\n",
       "      <td>44</td>\n",
       "      <td>142.0</td>\n",
       "      <td>49.0</td>\n",
       "      <td>Low value</td>\n",
       "      <td>1</td>\n",
       "      <td>Careful Spender</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>101</th>\n",
       "      <td>58</td>\n",
       "      <td>Female</td>\n",
       "      <td>19</td>\n",
       "      <td>127.0</td>\n",
       "      <td>15.0</td>\n",
       "      <td>Low value</td>\n",
       "      <td>4</td>\n",
       "      <td>Others</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102</th>\n",
       "      <td>68</td>\n",
       "      <td>Female</td>\n",
       "      <td>47</td>\n",
       "      <td>82.0</td>\n",
       "      <td>58.0</td>\n",
       "      <td>Medium value</td>\n",
       "      <td>1</td>\n",
       "      <td>Careful Spender</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>103</th>\n",
       "      <td>7</td>\n",
       "      <td>Female</td>\n",
       "      <td>50</td>\n",
       "      <td>127.0</td>\n",
       "      <td>99.0</td>\n",
       "      <td>High value</td>\n",
       "      <td>2</td>\n",
       "      <td>Luxury Seeker</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>100 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     CustomerID  Gender  Age  Annual_income  Spending_score Customer Type  \\\n",
       "0            78    Male   58          133.0            70.0  Medium value   \n",
       "1            34    Male   59           53.0             9.0     Low value   \n",
       "2            33    Male   43          142.0            33.0     Low value   \n",
       "3            88    Male   59          111.0            68.0  Medium value   \n",
       "4            80    Male   56           52.0            54.0     Low value   \n",
       "..          ...     ...  ...            ...             ...           ...   \n",
       "99           29    Male   44           17.0            27.0     Low value   \n",
       "100          75    Male   44          142.0            49.0     Low value   \n",
       "101          58  Female   19          127.0            15.0     Low value   \n",
       "102          68  Female   47           82.0            58.0  Medium value   \n",
       "103           7  Female   50          127.0            99.0    High value   \n",
       "\n",
       "     Cluster    Cluster_Label  \n",
       "0          1  Careful Spender  \n",
       "1          3           Others  \n",
       "2          4           Others  \n",
       "3          1  Careful Spender  \n",
       "4          0           Target  \n",
       "..       ...              ...  \n",
       "99         3           Others  \n",
       "100        1  Careful Spender  \n",
       "101        4           Others  \n",
       "102        1  Careful Spender  \n",
       "103        2    Luxury Seeker  \n",
       "\n",
       "[100 rows x 8 columns]"
      ]
     },
     "execution_count": 433,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 435,
   "id": "b1363729",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_csv(\"cleaned_customer_data.csv\", index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 436,
   "id": "fba2bfd7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-30 15:35:44.120 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Admin\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "# Load your cleaned dataset\n",
    "df = pd.read_csv(\"cleaned_customer_data.csv\")\n",
    "\n",
    "# Select and scale features\n",
    "X = df[['Annual_income', 'Spending_score']]\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Train KMeans (same n_clusters used before)\n",
    "kmeans = KMeans(n_clusters=5, random_state=42)\n",
    "df['Cluster'] = kmeans.fit_predict(X_scaled)\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"Customer Segmentation Predictor\")\n",
    "\n",
    "income = st.number_input(\"Enter Annual Income (in $K)\", min_value=0.0)\n",
    "score = st.number_input(\"Enter Spending Score (1–100)\", min_value=0.0, max_value=100.0)\n",
    "\n",
    "if st.button(\"Predict Segment\"):\n",
    "    # Scale input\n",
    "    user_data = scaler.transform([[income, score]])\n",
    "    prediction = kmeans.predict(user_data)[0]\n",
    "    \n",
    "    # Optional: add your own logic for labels\n",
    "    labels = {\n",
    "        0: \"Mid-Range Active\",\n",
    "        1: \"Upsell Candidate\",\n",
    "        2: \"Top Customer\",\n",
    "        3: \"Low Engagement\",\n",
    "        4: \"Luxury Dormant\"\n",
    "    }\n",
    "\n",
    "    st.success(f\"🎯 Predicted Cluster: {prediction} - {labels[prediction]}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "dc79fc7e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
