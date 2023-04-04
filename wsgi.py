from app import app, camera_processor_thread, camera_recording_thread



if __name__ == '__main__':
    app.run()
    camera_processor_thread.stop()
    camera_recording_thread.stop()
    
