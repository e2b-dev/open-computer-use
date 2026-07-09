import time
import threading
from multiprocessing import Process, Queue
import webbrowser
import subprocess
import os


class Browser:
    def __init__(self):
        self.width = 1024
        self.height = 768
        self.window_frame_height = 29  # Additional px for window border
        self.command_queue = Queue()
        self.webview_process = None
        self.is_running = False

    def open(self, url, width=None, height=None):
        """
        Open a browser window with the given URL

        Args:
            url (str): The URL to open
            width (int, optional): Window width
            height (int, optional): Window height
        """
        if self.is_running:
            print("Browser window is already running")
            return

        self.width = width or self.width
        self.height = height or self.height

        print(f"URL: {url}")
        
        # Try to open in system browser instead of pywebview
        try:
            # First try to open with the default browser
            success = webbrowser.open(url)
            if success:
                print("Opened VNC client in system browser")
                self.is_running = True
            else:
                print("System browser failed to open URL")
                raise RuntimeError("webbrowser.open() returned False")
        except Exception as e:
            print(f"Failed to open in system browser: {e}")
            
            # Fallback: try specific browsers
            browsers = ['firefox', 'chromium-browser', 'google-chrome', 'chrome']
            for browser in browsers:
                try:
                    subprocess.Popen([browser, url])
                    print(f"Opened VNC client in {browser}")
                    self.is_running = True
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"Failed to open {browser}: {e}")
                    continue
            
            if not self.is_running:
                print("Failed to open browser. Please manually navigate to:")
                print(url)

    def close(self):
        """Close the browser window"""
        if not self.is_running:
            print("No browser window is running")
            return

        # Since we're using system browser, we can't programmatically close it
        print("Please manually close the browser window")
        self.is_running = False

    @staticmethod
    def _create_window(url, width, height, command_queue):
        """This method is no longer used but kept for compatibility"""
        pass
