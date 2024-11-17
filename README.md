

# [SDBX](https://github.com/darkshapes/sdbx) / Powerful ML

<div align="center"> 
 
![the logo for 'shadowbox'. The word is written in a futuristic block typeface beneath a graffiti representation of the letters 's d b x' and the 'anomaly' logo for 'singuarlity'](https://github.com/user-attachments/assets/8913c77a-8252-4b18-8fc8-4160d6065cf7)
##

 ### [ [Darkshapes](https://github.com/darkshapes) ]
[ [Discord](https://discord.gg/RYaJw9mPPe) | [HuggingFace](https://huggingface.co/darkshapes) | [Maxtretikov](https://github.com/Maxtretikov/) | [EXDYSA](https://github.com/exdysa/) ]

</div>

<hr>

<div align="left">

## Windows Installation procedure for the dev branch

(contributed by mgua, (Marco Guardigli, mgua@tomware.it))

### Requirements:
- Microsoft Windows 64 bit
- Nvidia GPU, with recent video drivers supporting CUDA 12
- Microsoft Powershell
- Python, as installed from python.org, with py launcher
- minimum 8Gb of available space on your local storage
- git tool to be used from command line

It is recommended to use a non-admin user account to run this software.
The installation procedure may require you to approve/confirm setup from an admin.
A robust internet connection is recommended for the installation steps. 
(Approx 5 Gb are to be downloaded)


### Procedure
- In file explorer, enable visualization of hidden filesystem items
- Install python 3.11 or later, with py launcher
    https://www.python.org/downloads/release/python-3127/ 
- Install git for windows (possibly from https://git-scm.com/ )
- Open a powershell prompt and execute these commands one by one:



#### Creation of the python virtual environment venv_sdbx
This is where the libraries and packages are being downloaded and installed

A virtual environment is a python software execution context. 
It allows to isolate python projects minimizing conflicts and unwanted interactions.
Basically it is a folder in your provile. Its size growns when you install packages.

When you "activate" it, its specific version of python becomes the main one, 
with all the packages and libraries you installed. 
If not active, a python environment does nothing, beside eating disk space.

We create a new virtual environment, launching a powershell command prompt 
and executing the py launcher, to specify which python version we want to be mapped 
to "python" executable name in this virtual environment.

```
cd ~
py -3.12 -m venv venv_sdbx

```

We activate the newly created environment:
```
cd ~
./venv_sdbx/Scripts/Activate.ps1

```

Once the environment is active, the command line prompt shows the 
currently active environment name.

We proceed to install the most recent pip version in the environment.
pip (aka Python Install Manager) takes care of downloading and installing packages,
respecting and managing the cross-dependencies.

```
python -m pip install pip --upgrade

```


#### SDBX repository cloning
With this step we download from github the sdbx software. 
This will go in its own folder, separated from the previously created environment.

We create a folder and name it sdbx. 

This separation allows the project software to be "clean" from installation details
since sdbx can run on different operating systems, the os dependencies and libraries 
are within the venv_ folder.

The following git command downloads the shadowbox dev branch code from github, 
placing it in the new folder.

Future execution of _git pull_ commands from this folder will download updates. 

```
cd ~
mkdir sdbx
cd sdbx
git clone --branch dev https://github.com/darkshapes/sdbx.git .

```


#### SDBX installation
From a virtual environment activated prompt, we use pip to perform the installation, 
following the instructions included in the downloaded repository.
This steps takes a while, downloads approx 4.5Gbytes, and prepares the compiled version
of various components, for the specific platform you are using.
You may see warnings, but it should complete without troubles if you have enough available disk space.

```
cd ~
cd sdbx
python - m pip install -e .

```


#### Post-installation steps
The installation should take care of everything, but an additional step is needed.
This will be fixed in the future, but currently a simple file copy operation is needed to 
make sdbx work.

```
copy ~\sdbx\config\users\extensions.toml ~\Appdata\Local\Shadowbox

```



#### Running SDBX
Now we can run sdbx, running the following powershell command from its folder, 
having activated its virtual environment

```
cd ~
./venv_sdbx/Scripts/activate.ps1
sdbx -v
```

sdbx runs its server in a command prompt window, and spawns a browser, 
opening its own local address.


<hr>

### Visit the [Wiki](https://github.com/darkshapes/sdbx/wiki) for more information.

</div>
