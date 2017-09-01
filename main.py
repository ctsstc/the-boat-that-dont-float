from speedml import Speedml
sml = Speedml('../input/train.csv',
              '../input/test.csv',
              target='Survived',
              uid='PassengerId')
# sml.shape()
# sml.eda()
# sml.plot.correlate()
# sml.plot.distribute()
# sml.plot.continuous('Age')
# sml.plot.continuous('Fare')
sml.feature.outliers('Fare', upper=98)
# sml.plot.continuous('Fare')
# sml.plot.strip('Pclass', 'Fare')
# 
# sml.plot.ordinal('SibSp')
sml.feature.outliers('SibSp', upper=99)
# sml.plot.ordinal('SibSp')
# sml.plot.strip('SibSp', 'Age')
# 
# sml.eda()

sml.feature.density('Age')
sml.train[['Age', 'Age_density']].head()

sml.feature.density('Ticket')
sml.train[['Ticket', 'Ticket_density']].head()

# Shows to be low density, let's scrap this!
sml.feature.drop(['Ticket'])

# sml.plot.crosstab('Survived', 'SibSp')
# sml.plot.crosstab('Survived', 'Parch')

# Add some new features, and transform data
sml.feature.fillna(a='Cabin', new='Z')
sml.feature.extract(new='Deck', a='Cabin', regex='([A-Z]){1}')
sml.feature.drop(['Cabin'])
sml.feature.mapping('Sex', {'male': 0, 'female': 1})
sml.feature.sum(new='FamilySize', a='Parch', b='SibSp')
sml.feature.add('FamilySize', 1)

#sml.plot.bar('FamilySize', 'Survived')
#sml.plot.bar('Deck', 'Survived')
# Drop old fields now that we have new ones
sml.feature.drop(['Parch', 'SibSp'])

# Fill in missing values with median values
# ie: There are plenty of Cabin/Deck values missing from the data
sml.feature.impute()

#sml.info()
#sml.plot.importance()
sml.train.head()

sml.feature.extract(new='Title', a='Name', regex=' ([A-Za-z]+)\.')
sml.plot.crosstab('Title', 'Sex')