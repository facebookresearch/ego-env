# Datasets

If not already done, download data, checkpoints and annotations.
```
bash data/download_data.sh
```

### HouseTours
**Download HouseTours videos** to `data/housetours/videos/`. Follow [instructions](https://github.com/uiuc-robovision/video-dqn?tab=readme-ov-file#building-dataset) from *Semantic Visual Navigation by Watching Youtube Videos* (NeurIPS 20).

**Generate video clips** from videos corresponding to segments that were annotated. Output: `data/housetours/clips/*.mp4`
```
python -m data.generate_housetours_clips
```

Annotations for RoomPred and NLQ are in `data/annotations/` and reference clips generated above. Example RoomPred annotations:

|           clip_uid | start_time |  end_time |                label |
|-------------------:|-----------:|----------:|---------------------:|
| D-u5F73q1p8_75_383 |  134.61400 | 146.54800 |     corridor/hallway |
| zqMCzbfZIP0_16_188 |   30.01252 |  38.02866 | office / home_office |
|  y4Ls5f5yFsk_2_285 |  131.72732 | 131.72732 |          living_room |
| yQJDlZqedbQ_60_197 |   30.20826 |  54.91000 |          dining_room |
|  s79QloFoSPA_1_270 |   18.59712 |  25.47814 |  front_door/entrance |

Example NLQ annotations:
|            clip_uid |                                                  query | response_start | response_end |          category |
|--------------------:|-------------------------------------------------------:|---------------:|-------------:|------------------:|
|  zmiUF_MdkoY_55_223 |                when did i visit gas cooker in kitchen? |          258.0 |        294.0 |      visit_x_in_y |
|  bpxOMabtssg_28_287 |     when did i go from the dining room to the kitchen? |          170.0 |        268.0 | visit_room_xtheny |
|   y4Ls5f5yFsk_2_285 |             where did i see a car in the parking lot ? |            0.0 |         34.0 |        see_x_in_y |
|  3lMpvKbW4x8_28_205 | where did i first see a floor lamp in the living room? |          114.0 |        144.0 |        see_x_in_y |
| XSB5KE1OI9Q_166_345 |     when did i go from the kitchen to the living room? |           90.0 |        146.0 | visit_room_xtheny |

### Ego4D
**Download Ego4D clips and annotations** using the [official CLI tool](https://ego4d-data.org/docs/CLI/). Annotations for RoomPred are in `data/annotations/` and reference clips downloaded using CLI. Example RoomPred annotations:

|                            video_uid |                             clip_uid | start_time |   end_time |       label | instance |
|-------------------------------------:|-------------------------------------:|-----------:|-----------:|------------:|---------:|
| cf7c12db-1a9e-46d3-96d6-38174bbe373c | 3720b579-22a1-4ca0-be37-0b1018ab765f | 2400.00000 | 3461.87000 | living_room |        0 |
| 511b0123-7b3d-4d3f-a0cf-8c3ca3490fc6 | 4b05abcc-f252-416b-89d9-63df3dec94f0 |   54.91008 |  363.74236 | living_room |        0 |
| 38a7b760-56f9-4565-8b70-f8dad5768ace | 348cf9e3-c75e-49ec-8ae4-8562d8b4bfd1 |  790.20567 |  927.81003 |     kitchen |        0 |
| eff7f167-e828-421f-a69e-956dddbecf08 | b7982557-87df-4fe8-b7cb-e86ca1c1d21c |  133.35846 |  136.11082 |     kitchen |        0 |
| b3190c93-c6fc-4e3d-b6ca-41e7ea9fed3f | ad7aadad-462b-4003-bb42-592114d9364c | 3181.96000 | 3536.25300 |     bedroom |        1 |

Note: The instance id distinguishes multiple rooms of the same category (e.g., bedroom 1 vs. bedroom 2). This is not used in our experiments, but provided as extra metadata.

For NLQ, use the official Ego4D NLQ challenge annotations. See the [challenge documentation](https://eval.ai/web/challenges/challenge-page/1629/overview) for more information. Features are generated at 1FPS and incorporated directly into [prior work](https://github.com/srama2512/NaQ). Pre-computed EgoEnv features for Ego4D NLQ videos can be downloaded [here](https://dl.fbaipublicfiles.com/ego-env/data/ego4d_egoenv_nlq_feats.zip).