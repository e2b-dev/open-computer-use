from e2b_desktop import Sandbox as SandboxBase
import asyncio
import os
import signal
import sys


class Sandbox(SandboxBase):

    def start_stream(self):
        # Command to start streaming using ffmpeg
        command = "ffmpeg -f x11grab -s 1024x768 -framerate 30 -i {self._display} -vcodec libx264 -preset ultrafast -tune zerolatency -f mpegts -listen 1 http://localhost:8080"
        # Run the command in the background
        process = self.commands.run(
            command,
            background=True,
        )
        self.process = process
        return f"https://{self.get_host(8080)}"

    def kill(self):
        # Kill the streaming process along with the sandbox
        if hasattr(self, "process"):
            self.process.kill()
        super().kill()


# Client to view and save a live display stream from the sandbox
class DisplayClient:
    def __init__(self, output_dir="."):
        self.process = None
        # Define output stream and file paths
        self.output_stream = f"{output_dir}/output.ts"
        self.output_file = f"{output_dir}/output.mp4"

    async def start(self, stream_url, title="Sandbox", delay=0):
        title = title.replace("'", "\\'")
        # Start a subprocess to both view and save the stream
        self.process = await asyncio.create_subprocess_shell(
            f"sleep {delay} && ffmpeg -reconnect 1 -i {stream_url} -c:v libx264 -preset fast -crf 23 "
            f"-c:a aac -b:a 128k -f mpegts -loglevel quiet - | tee {self.output_stream} | "
            f"ffplay -autoexit -i -loglevel quiet -window_title '{title}' -",
            preexec_fn=os.setsid,
            stdin=asyncio.subprocess.DEVNULL,
        )

    async def stop(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            await self.process.wait()

    async def save_stream(self):
        # Convert the saved stream to an mp4 file
        process = await asyncio.create_subprocess_shell(
            f"ffmpeg -i {self.output_stream} -c:v copy -c:a copy -loglevel quiet {self.output_file}"
        )
        await process.wait()

        if process.returncode == 0:
            print(f"Stream saved successfully as {self.output_file}.")
        else:
            print(f"Failed to save the stream as {self.output_file}.")
