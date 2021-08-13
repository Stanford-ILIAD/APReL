# Installation
**APReL** runs on Python 3.

### Install Requirements & Run
1. Clone the robosuite repository
```sh 
$ git clone https://github.com/Stanford-ILIAD/APReL.git
$ cd APReL
```

2. Install the base requirements with
   ```sh
   $ pip3 install -r requirements.txt
   ```
   NOTE: **APReL** is not installed as an independent package to your system. Instead, you can move the _aprel_ folder (after you install the base requirements) to import it as needed.

3. (Optional) If you want to build the docs locally, you will also need some additional packages, which can be installed with:
   ```sh
   $ pip3 install -r requirements-docs.txt
   ```

4. Test **APREL**'s runner file by running
   ```sh
   $ python run.py --env "MountainCarContinuous-v0" --max_episode_length 100 --num_trajectories 10
   ```
   You should be able to see the [MountainCarContinuous-v0](https://gym.openai.com/envs/MountainCarContinuous-v0/) environment rendering multiple times. After it renders (and saves) 10 trajectories, it is going to query you for your preferences. See the [Example page](example.html) for more information about this runner file.