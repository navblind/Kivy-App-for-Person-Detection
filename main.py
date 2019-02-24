from kivy.app import App
from kivy.uix.label import Label
import cv2


class CamApp(App):
    def build(self):
        cap = cv2.VideoCapture(1)
        cap.release()
        
        return Label(text="hello")


if __name__ == '__main__':
    CamApp().run()
