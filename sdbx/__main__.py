import uvicorn
import webbrowser

from sdbx import config

def main():
	if config.web.auto_launch:
		webbrowser.open(f"http://{config.web.listen}:{config.web.port}", new=0, autoraise=True)

	uvicorn.run("sdbx:app", host=config.web.listen, port=config.web.port, reload=True)

if __name__ == "__main__":
	main()