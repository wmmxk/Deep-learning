1. The compability of ubuntu 16.40 with third party drivers:
   After searching GTX 1070 Ubuntu 16.04, I found this blog is very informative except without mentioning this compability issue. (https://www.jayakumar.org/linux/gtx-1070-on-ubuntu-16-04-with-cuda-8-0-and-theano/)
   This issue is because UEFI Secure Boot is enabled by default sind 16.04, so you need to disable Secure Boot or assign a certificate to the nvidia driver. (I chose the former solution).
   My desktop is Msi. Go to BIOS seting by pressing "del" key when rebooting and disable the secure booting



2. The compability of the gcc coming with ubuntu 16.40 with CUDA 8.0
   It looks like the gcc is not compatible with CUDA 8.0. Since I have root right, I just install another compiler and create a symbolic link and covered the default gcc
   sudo apt-get install gcc-4.9 g++-4.9
   sudo ln -s  /usr/bin/gcc-4.9 /usr/bin/gcc -f
   sudo ln -s  /usr/bin/g++-4.9 /usr/bin/g++ -f
   
   unlink a symbolic link:
   unlink /usr/bin/gcc

3. When installing CUDA 8.0, make sure say yes when being asked whether create symbolic link. I think that way no matter which version CUDA you installed, it will use cuda to refer the CUDA you install.

4.It was said to install nvidia driver manually. (Note the 1st compability issue surfaced when I run sudo apt-get install nvidia-367. (A new window pop out saying "Your system has UEFI Secure Boot enabled. UEFI Secure Boot is not compatible with the use of third-party drivers.
(...) Ubuntu will still be able to boot on your system but these third-party drivers will not be available for your hardware."

When rebooting, **I disabled the Secure boot**. On May 23, 2019, you enabled something then it worked. Otherwise, I can not log in: everytime I typed in the password for login, it immediately throws me back to the login screen.



I installed the driver for GTX 1070 in the following way:
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt-get update
   sudo apt-get install nvidia-367
   sudo reboot

5. After I installed CUDA, I need to add the directory for executive file and libraries to PATH and LD_LIBRARY_PATH, my vim editor does not work as expected. (The backspace, arrow up, down keys do not work)
   Solution:
   create a .vimrc file in your home directory
   type the following configuration inside:
      set nocompatible
      set backspace=indent,eol,start
      set nubmer
      
     

5.2. Install cuda9.0 following [this tutorial](https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73)

5.3 Install cuda10: follow this [link](https://www.pugetsystems.com/labs/hpc/How-To-Install-CUDA-10-together-with-9-2-on-Ubuntu-18-04-with-support-for-NVIDIA-20XX-Turing-GPUs-1236/). In the 2nd step, change the <version> or just run the recommended command from the prompt.
   If you need to confirm the version of your ubuntu, run lsb_release -a
   After you install cuda10, go to /usr/local, run ls -la to show the real path of the symbolic link, cuda
   If you can not run nvidia-smi after you install cuda10, just reboot your computer.
   Still you need to add the cuda path to PATH and LD_LIBRARY_PATH, if they are not. To check this just run echo $PATH |grep cuda, echo $LD_LIBRARY_PATH | grep cuda.
   

6. How to solve gcc compabiltiy issue, you could change the configuration of CUDA rather than install gcc 4.9. But it is more complicated.

7. The installation of cudnn:
   After you install cuda, download the cudnn library
   tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz
   cd cuda # The name of uncompresed foder is cuda not cunn
   sudo cp -P /include/cudnn.h /usr/include
   sudo cp -P /lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
   sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*

8. After the driver, cuda, cudnn, vi are setup, then I installed anaconda and pycharm.
   For pycharm.sh, I created an alias for it by adding this line in the .bashrc file:
   alias pycharm="/home/wxk/software/pycharm-edu-2019.1/bin/pycharm.sh"
