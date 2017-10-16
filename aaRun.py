from createobservation import createobservation
from inversemethod import inversemethod

# ;aaRun.pro
print('test') 


s = createobservation( 'hd209458_100' ) 
s = createobservation( 'hd209458_200' ) 
s = createobservation( 'hd209458_300' ) 
s = createobservation( 'hd209458_400' ) 

x = inversemethod( 'test' , 10 ) 
x = inversemethod( 'hd209458_100' , 100 ) 
x = inversemethod( 'hd209458_100' , 1000 ) 
x = inversemethod( 'hd209458_100' , 10000 ) 

x = inversemethod( 'hd209458_200' , 10 ) 
x = inversemethod( 'hd209458_200' , 100 ) 
x = inversemethod( 'hd209458_200' , 1000 ) 
x = inversemethod( 'hd209458_200' , 10000 ) 

x = inversemethod( 'hd209458_300' , 10 ) 
x = inversemethod( 'hd209458_300' , 100 ) 
x = inversemethod( 'hd209458_300' , 1000 ) 
x = inversemethod( 'hd209458_300' , 10000 ) 

x = inversemethod( 'hd209458_400' , 10 ) 
x = inversemethod( 'hd209458_400' , 100 ) 
x = inversemethod( 'hd209458_400' , 1000 ) 
x = inversemethod( 'hd209458_400' , 10000 ) 
