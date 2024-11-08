# kanimtool

Process kanim files in Oxygen Not Included.

## Examples

This command creates Hatch's idle animation in `hatch.webp`:

```shell
kanimtool-make \
  --build hatch_build.bytes --texture hatch_0.png \
  --map snapto_pivot= \
  hatch_anim.bytes idle_loop hatch.webp
```

Stone hatch

```shell
kanimtool-make \
  --build hatch_build.bytes --texture hatch_0.png \
  --map snapto_pivot= --affix hvy_ \
  hatch_anim.bytes idle_loop stone_hatch.webp
```

Cheers from Freyja!

```shell
kanimtool-make \
  --build body_comp_default_build.bytes --texture body_comp_default_0.png \
  --build head_swap_build.bytes --texture head_swap_0.png \
  --build hair_swap_build.bytes --texture hair_swap_0.png \
  --build body_swap_build.bytes --texture body_swap_0.png \
  --map snapto_headfx=,snapto_rgthand=,snapto_goggles=,snapto_hat=,snapto_hat_hair= \
  --map skirt=,snapto_chest=,snapto_neck=,necklace= \
  --map snapto_headshape=headshape_001 \
  --map snapto_cheek=cheek_001 \
  --map snapto_eyes=eyes_002 \
  --map snapto_mouth=mouth_001 \
  --map snapto_hair=hair_038 \
  --map torso=torso_002 \
  --map arm_sleeve=arm_sleeve_002 \
  --map arm_lower_sleeve=arm_lower_sleeve_002 \
  --map arm_lower=arm_lower_001 \
  --map arm_upper=arm_upper_001 \
  anim_cheer_anim.bytes cheer_pre,cheer_loop,cheer_pst cheer.webp
```

Inspect build and kanim files
```shell
kanimtool-inspect hatch_build.bytes
kanimtool-inspect hatch_anim.bytes
```

##
This project is a modding tool of Oxygen Not Included and is not associated with Klei.
Oxygen Not Included and related data are Klei Entertainment properties.
