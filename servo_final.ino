
#include <SPI.h>
#include <ServoCds55.h>
ServoCds55 myservo;

int servoNum = 1;
String inString = "";         // a String to hold incoming data
boolean stringComplete = false;  // whether the string is complete
int error;
float Kp = -0.29;
int vel = 0;

void setup() {
  // initialize serial:
  Serial.begin(115200);
  // reserve 20 bytes for the inputString:
  inString.reserve(20);
  myservo.begin ();
  myservo.Reset(servoNum);
  myservo.setVelocity(10);// set velocity to 100(range:0-300) in Servo mode
  Serial.print ("Ready...\n");
}

void loop() {
  // print the string when a newline arrives:
  if (stringComplete) {
    
    // clear the string:
    vel = Kp * error;
    Serial.println(vel);
    myservo.rotate(servoNum, vel); //   Anti CW    ID:1  Velocity: 150_middle velocity  300_max
    inString = "";
    stringComplete = false;
  }
}

void serialEvent() {

  while (Serial.available() > 0)
  {
    int inChar = Serial.read();
    if ((isDigit(inChar))||(inChar == '-')) {
      // convert the incoming byte to a char and add it to the string:
      inString += (char)inChar;
    }
    // if you get a newline, print the string, then the string's value:
    if (inChar == 'p')
    {
      //      Serial.print("Value:");
      //      Serial.println(inString.toInt());
      //      Serial.print("String: ");
      //      Serial.println(inString);
      // clear the string for new input:
      error = inString.toInt();
      inString = "";
      stringComplete = true;
    }
  }
}
