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

# Shows to be low density, let's scrap this!
sml.feature.drop(['Ticket'])


