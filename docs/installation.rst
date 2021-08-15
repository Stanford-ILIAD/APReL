Installation
########

**APReL** runs on Python 3.

Install from Source
**********************

1. **APReL** uses `ffmpeg <https://www.ffmpeg.org/>`_ for trajectory visualizations. Install it with the following command on Linux:

.. code-block:: sh

   apt install ffmpeg

If you are using a Mac, you can use `Homebrew <https://brew.sh/>`_ to install it:

.. code-block:: sh

   brew install ffmpeg


2. Clone the aprel repository

.. code-block:: sh

   git clone https://github.com/Stanford-ILIAD/APReL.git
   cd APReL


3. Install the base requirements with

.. code-block:: sh

   pip3 install -r requirements.txt


4. (Optional) If you want to build the docs locally, you will also need some additional packages, which can be installed with:

.. code-block:: sh

   pip3 install -r docs/requirements.txt


5. Install **APReL** from the source by running:

.. code-block:: sh

   pip3 install -e .


6. Test **APReL**'s runner file by running

.. code-block:: sh

   cd examples
   python simple.py


You should be able to see the `MountainCarContinuous-v0 <https://gym.openai.com/envs/MountainCarContinuous-v0/>`_ environment rendering multiple times.
After it renders (and saves) 10 trajectories, it is going to query you for your preferences. See the next section for more information about this runner file.