====================================
APReL: A Library for Active Preference-based Reward Learning Algorithms
====================================

.. image:: https://readthedocs.org/projects/aprel/badge/?version=latest
  :target: http://aprel.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

**APReL** is a unified Python3 library for active preference-based reward learning methods. It offers a modular framework for experimenting with and implementing preference-based reward learning techniques; which include active querying, multimodal learning, and batch generation methods.


Installation
########

**APReL** runs on Python 3.

Install Requirements & Run
**********************

1. Clone the robosuite repository
.. code-block:: sh

   git clone https://github.com/Stanford-ILIAD/APReL.git
   cd APReL


2. Install the base requirements with
.. code-block:: sh

   pip3 install -r requirements.txt


NOTE: **APReL** is not installed as an independent package to your system. Instead, you can move the _aprel_ folder (after you install the base requirements) to import it as needed.

3. (Optional) If you want to build the docs locally, you will also need some additional packages, which can be installed with:
.. code-block:: sh

   pip3 install -r docs/requirements.txt


4. Test **APREL**'s runner file by running
.. code-block:: sh

   pip3 install -r requirements.txt


You should be able to see the `MountainCarContinuous-v0 <https://gym.openai.com/envs/MountainCarContinuous-v0/>`_ environment rendering multiple times. After it renders (and saves) 10 trajectories, it is going to query you for your preferences. See the next section for more information about this runner file.


Example
########

Under construction.