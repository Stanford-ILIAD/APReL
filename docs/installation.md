# Installation
**APReL** runs on Python 3.

### Install from Source
1. **APReL** uses [ffmpeg](https://www.ffmpeg.org/) for trajectory visualizations. Install it with the following command on Linux:
   ```sh 
   $ apt install ffmpeg
   ```

   If you are using a Mac, you can use [Homebrew](https://brew.sh/) to install it:
   ```sh 
   $ brew install ffmpeg
   ```

2. Clone the robosuite repository
   ```sh 
   $ git clone https://github.com/Stanford-ILIAD/APReL.git
   $ cd APReL
   ```

3. Install the base requirements with
   ```sh
   $ pip3 install -r requirements.txt
   ```

4. (Optional) If you want to build the docs locally, you will also need some additional packages, which can be installed with:
   ```sh
   $ pip3 install -r docs/requirements.txt
   ```

5. Install **APReL** from the source by running:
   ```sh
   $ pip3 install -e .
   ```

6. Test **APREL**'s runner file by running
   ```sh
   $ cd examples
   $ python run.py --env "MountainCarContinuous-v0" --max_episode_length 100 --num_trajectories 10
   ```
   You should be able to see the [MountainCarContinuous-v0](https://gym.openai.com/envs/MountainCarContinuous-v0/) environment rendering multiple times. After it renders (and saves) 10 trajectories, it is going to query you for your preferences. See the [Example page](example.html) for more information about this runner file.