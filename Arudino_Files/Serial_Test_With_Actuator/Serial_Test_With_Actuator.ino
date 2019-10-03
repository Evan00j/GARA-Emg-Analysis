/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 http://www.arduino.cc/en/Tutorial/Sweep
*/

#include <Servo.h>

Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 90;    // variable to store the servo position
int wind = 0;
void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT);
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
}

void loop() {

  if (Serial.read() == 'l'){
    digitalWrite(LED_BUILTIN, HIGH);
    if(pos == 60)
      wind = 0;
    if(pos == 120)
      wind = 1;
      if (wind == 0){
        for (int i=0; i <= 5; pos++){
      if(pos == 120)
      wind = 1;
      myservo.write(pos);      
      i++;// tell servo to go to position in variable 'pos'
      delay(15);
      }
    }else if (wind == 1){
      
      for (int i=0; i <= 5; pos--){
      myservo.write(pos);
      if(pos == 60)
      wind = 0;   
      i++;// tell servo to go to position in variable 'pos'
      delay(15);
      } 
      // waits 15ms for the servo to reach the position
  }
  /*
  for (pos = 180; pos >= 0; pos -= 1) { // goes from 180 degrees to 0 degrees
    myservo.write(pos);             // tell servo to go to position in variable 'pos'
    delay(15);                       // waits 15ms for the servo to reach the position
  }*/
  }
  else digitalWrite(LED_BUILTIN, LOW);
  
  
}
