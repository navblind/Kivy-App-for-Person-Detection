import kivy

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.label import Label

import cv2
from detection.objectDetection import Detect
import pyttsx3



camera = '''
BoxLayout:
    orientation: 'horizontal'
    Camera:
        resolution: (1920, 1080)
        id: Cam0
        play: True
'''



class myCamera(App):
    cap = cv2.VideoCapture(0)
    det = Detect()
    engine = pyttsx3.init()
    def build(self):
        while True:
            _, img = self.cap.read()
            people = self.det.run(img)

            self.engine.say(people[0])
            self.engine.runAndWait()


            print(people)
        label = Label(text="hello")
        #return Builder.load_string(camera)

        return label
if __name__=="__main__":
    myCamera().run()
