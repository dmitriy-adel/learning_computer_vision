import cv2

from models_usage import ObjectClassificationModel


class WebCameraViewer:
    def __init__(self):
        self.object_classification_model: ObjectClassificationModel = ObjectClassificationModel()
        self.web_camera_capture: cv2.VideoCapture = None
        self.capture_web_camera()

    def capture_web_camera(self):
        try:
            self.web_camera_capture = cv2.VideoCapture(0)
            if not self.web_camera_capture.isOpened():
                print("[WebCameraViewer->capture_web_camera]. Cant capture web_camera for some reason")
                exit()

        except Exception as _ex:
            print(f'[WebCameraViewer->capture_web_camera]. Cant capture web_camera. Error :: {_ex}')
            exit()

    def read_web_camera_image(self):
        while True:
            ret, frame = self.web_camera_capture.read()
            
            if not ret:
                print("[WebCameraViewer->read_web_camera_image]. cant get web camera frame")
                continue
            
            frame = self.object_classification_model.classify_image(frame=frame)
            cv2.imshow("Webcam", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def run(self):
        self.read_web_camera_image()


if __name__ == '__main__':
    wcv_capture: WebCameraViewer = WebCameraViewer()
    wcv_capture.run()

    cv2.destroyAllWindows()
