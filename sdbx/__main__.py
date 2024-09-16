import uvicorn
import webbrowser

from sdbx import config, logger, source

def main():
	if config.web.auto_launch:
		webbrowser.open(f"http://{config.web.listen}:{config.web.port}", new=0, autoraise=True)

	uvicorn.run(
		"sdbx:app",
		factory=True,
		
		host=config.web.listen, 
		port=config.web.port, 
		reload=config.web.reload,
		reload_dirs=[source, config.path],
		reload_includes=config.web.reload_include
	)

if __name__ == "__main__":
	main()