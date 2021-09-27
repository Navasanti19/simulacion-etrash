import time
import telepot
import cv2
import serial
import sys
global clientID
global device
global Puerto_Serie
global Arduino
global people
people=[]
def handle(msg):
    global people
    command=msg['text']
    person=int(msg['from']['id'])
    #print(person)
    people.append(person)
    if command=='Foto':
        bot.sendMessage(person, 'Última Foto del Recipiente')
        bot.sendPhoto(person,photo=open('objeto.jpg','rb'))

    elif command == '/start' or command == 'Start':
        bot.sendMessage(person, 'Bienvenid@ a E-Trash')
        bot.sendMessage(person, 'Podrás interactuar con la caneca por medio de los siguientes comandos: \n'
                                   '\'Foto\': Recibirás la última foto tomada por la caneca\n'
                                   '\'Ayuda\': Tendrás información para contactarte con el equipo creador\n'
                                   '\'Start\': Para repetir este menú')
        bot.sendMessage(person,'Además por este medio se te avisará cuándo una caneca interna esté llena para su opotuna recogida')

    elif command == 'Ayuda':
        bot.sendMessage(person, 'Veo que necesitas ayuda...')
        bot.sendMessage(person, 'Te ayudaré a contactar al equipo de trabajo para que soluciones tus dudas')
        bot.sendMessage(person, 'También puedes contactarte al correo: etrash@gmail.com')
        bot.sendMessage(person, 'Muchas gracias por contactarte!')

    else:
        bot.sendMessage(person, 'Comando no disponible \nIntenta Otro Comando...')

Puerto_Serie = serial.Serial('COM3',9600)
time.sleep(1)

control = False

#bot = telepot.Bot('1183214273:AAHgXQk-EXALGZRdQ5RM1klfUDxAp94gGcw')
bot = telepot.Bot('1956845867:AAF7hh76U0oFXVdh1u82EsA6r4OztU6GrKM')
bot.message_loop(handle)

print('Conectando...')

 
aviso=True
while 1:
    Ardu = Puerto_Serie.readline()
    Arduino=Ardu.decode('utf-8')
    Arduino=Arduino[0]
    
    #time.sleep(2)
    if Arduino == 'A' and aviso == True:
        #for i in range(len(people)):
            bot.sendMessage(902264668, 'Basura Orgánica llena')
            aviso=False
        
    elif Arduino == 'B'and aviso == True:
        #for i in range(len(people)):
            bot.sendMessage(902264668, 'Basura Plásticos llena')
            aviso=False
        
    elif Arduino == 'C'and aviso == True:
        #for i in range(len(people)):
            bot.sendMessage(902264668, 'Basura Cartón llena')
            aviso=False
    else:
        aviso=True

