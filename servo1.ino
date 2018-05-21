//myservo.Reset(servoNum);    //Only Dynamixel AX need this instruction while changing working mode
//CDS55xx don't need this, it can switch freely between its working mode

#include <SPI.h>
#include <ServoCds55.h>
ServoCds55 myservo;

int vel; //velocidad del servo
int error;   //error de centrado de la camara
int Kp = 1;
int servoNum = 1;
int inputCommand ;         // a string to hold incoming data


void setup ()
{
  Serial.begin (115200);
  Serial.print ("Ready...\n");
  myservo.begin ();
  myservo.Reset(servoNum);//Restore ID2 servo to factory Settings ( ID:1  Baud rate:1000000)
}

void loop ()
{
  serialEvent();
  controlServo(error);
}


void serialEvent()
{
  while (Serial.available())
  {
    error = Serial.read();
  }
}

void controlServo(char val)
{

  vel = error * Kp;
  myservo.rotate(servoNum, vel); //  CW     ID:1  Velocity: -150_middle velocity  -300_max
  delay(1000);
}

  




