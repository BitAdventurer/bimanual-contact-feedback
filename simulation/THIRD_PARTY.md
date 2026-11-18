# Simulation asset provenance

The portable scene references 66 external files from the public **robosuite 1.4.1** wheel. They comprise Panda arm meshes, Panda finger/hand meshes, the Rethink pedestal and two arena textures. The repository contains their relative filenames and SHA-256 values. Installation obtains the original asset bytes from the upstream distribution. No mesh or texture binaries are included here. [Versioned release](https://pypi.org/project/robosuite/1.4.1/)

The combined XML scene includes robot, gripper and arena descriptions derived from robosuite, together with the study's task geometry and material settings. The upstream release carries the MIT license and notes that its partial DeepMind MuJoCo implementation is licensed under Apache-2.0. The versioned robosuite license is reproduced in [ROBOSUITE-LICENSE.txt](ROBOSUITE-LICENSE.txt). [Upstream license](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/LICENSE)

MuJoCo is used via its Python binary distribution, which supplies its own license and third-party notices. [MuJoCo 3.2.3 license](https://github.com/google-deepmind/mujoco/blob/3.2.3/LICENSE)
