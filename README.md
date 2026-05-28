# ZK Task Planner

A vision-grounded task planner and skill-generation system for real robots. The framework takes a natural-language task ("open the bottle", "set a place at the table"), looks at the scene through an RGB-D camera, and produces a sequence of executable motion primitives on an xArm with CuRobo motion planning — no hand-coded skill library required.

The core contribution is a **two-stage skill system**: a vision-language model proposes a symbolic *skill* (parameterized primitives over points-of-interest on the object), and a deterministic *handler* instantiates that skill against the live 3D scene to produce a robot trajectory.

---

## Pipeline overview

```
              natural-language task + RGB image
                            │
                            ▼
                      TaskPlanner
                  (LLM task decomposition)
                            │
                  skill commands, e.g. open(bottle)
                            │
                            ▼
                  DirectSkillExecutor
            ┌───────────────┴───────────────┐
            ▼                               ▼
      PerceptionSystem            SkillGenerator / SkillHandler
   YOLO-World + FastSAM           ┌─────────────────────────────┐
   object mask + POIs             │ 1. find_similar_skill (LLM) │
   surface normals                │ 2. generate_skill (VLM)     │
            │                     │ 3. adapt_skill (VLM)        │
            │                     └──────────────┬──────────────┘
            │                                    │
            │            symbolic Skill (primitives + POI labels)
            │                                    │
            └─────────────► PrimitiveParser ◄────┘
                                    │
                                    ▼
                          ExecutableAction sequence
                       (3D poses, gripper commands)
                                    │
                                    ▼
                       CuRoboMotionPlanner → xArm
```

---

## Skill generation, in detail

The heart of the system is the **skill** abstraction in [skill_generator.py:32-46](cognitive_bt_framework/src/skills/skill_generator.py#L32-L46):

```python
@dataclass
class Skill:
    name: str
    abstract_action: str          # e.g. "open"
    target_object: str            # e.g. "bottle"
    primitive_sequence: List[str] # ["move_gripper_to_pose('A', true, false)",
                                  #  "close_gripper()",
                                  #  "twist('A', 90)", ...]
    parameters: Dict[str, Any]
    prerequisites: List[str]
    constraints: List[str]
    image_id: str
    points_of_interest: Dict[str, PointOfInterest]  # labels 'a','b','c',... → 2D positions
    explanations: List[str]
    surface_info: Optional[Dict[str, Any]] = None
    object_bbox: Optional[List[int]] = None
```

A skill is **symbolic, not numeric**. The primitive sequence references *points of interest* by alphabetical label (`'A'`, `'B'`, ...) rather than 3D coordinates. The same skill — same primitive sequence, same point labels — can be reused for any object where the perception system finds analogous points. This is what makes the skill library transferable.

### The three skill operations

[`SkillGenerator`](cognitive_bt_framework/src/skills/skill_generator.py#L48) exposes three operations against the LLM, called from [`SkillHandler.instantiate_skill`](cognitive_bt_framework/src/skills/skill_handler.py#L81):

#### 1. `find_similar_skill` — retrieval
[skill_generator.py:1878](cognitive_bt_framework/src/skills/skill_generator.py#L1878)

Given the new object image + POIs and the requested `(abstract_action, target_object)` pair, query the VLM with the new view alongside *every cached skill image* for that action/object combination. The VLM returns a similarity score in `[0, 1]` per cached skill; if the best score exceeds `0.8`, the cached skill is reused.

#### 2. `generate_skill` — synthesis
[skill_generator.py:1240](cognitive_bt_framework/src/skills/skill_generator.py#L1240)

If no similar skill exists, the VLM is shown:
- The RGB image with **POIs overlaid as alphabetical labels**.
- A separate visualization with the **object mask outlined** and arrows labeling its `top`/`bottom`/`left`/`right` edges.
- Optional **surface normals** rendered from the depth image.
- A textual description of POIs, surfaces, action, and any previously executed skills (for context).

It is asked to output a primitive sequence using only labeled points — e.g. `push('A', perpendicular, false, true, 'top')`. The output is parsed, stored on disk as a `Skill` JSON, and cached in memory.

#### 3. `adapt_skill` — transfer
[skill_generator.py:1984](cognitive_bt_framework/src/skills/skill_generator.py#L1984)

When `find_similar_skill` finds a close-but-not-exact match, `adapt_skill` shows the VLM both the base skill image and the new object's POIs and asks it to remap the primitive sequence's point labels to the new object's POI labels. Cheaper than full regeneration and preserves the symbolic structure.

### Points of interest

POIs are produced upstream by the perception system ([interaction_point_detector_v2.py](cognitive_bt_framework/src/vision/interaction_point_detector_v2.py)) — a geometric detector that finds graspable/pushable/pullable points on the object mask using contour analysis, depth discontinuities, and surface-normal estimation. Each POI carries position, detection method, interaction type, and stability/accessibility scores ([skill_generator.py:17-29](cognitive_bt_framework/src/skills/skill_generator.py#L17-L29)).

The VLM never sees a 3D coordinate. It reasons purely over the labeled image, which keeps the prompt scale-invariant and lets the same skill transfer across object instances.

### Primitive vocabulary

[`PrimitiveParser`](cognitive_bt_framework/src/skills/primitive_parser.py#L12) accepts the following primitives (each in multiple syntactic forms: positional, named-keyword, and legacy keyword-based):

| Primitive | Purpose |
|---|---|
| `move_gripper_to_pose('A', is_top_down_grasp, is_side_grasp)` | Move TCP to POI `A` with the specified approach |
| `push('A', force_direction, is_button, has_pivot, hinge_location)` | Push at POI `A`, optionally around a hinge |
| `pull('A', force_direction, is_button, has_pivot, hinge_location)` | Pull at POI `A`, optionally around a hinge |
| `twist('A', angle)` | Rotate the gripper around POI `A` |
| `open_gripper()` / `close_gripper()` | Gripper control |
| `retract_gripper()` | Safe retract to home |

Surface-keyword variants (e.g. `push([surface_keywords], is_parallel_surface=...)`) are retained for backward compatibility.

### Instantiation: symbol → trajectory

[`SkillHandler._instantiate_skill`](cognitive_bt_framework/src/skills/skill_handler.py#L216) walks the symbolic primitive sequence and, for each step:

1. Parses the primitive with `PrimitiveParser`.
2. Resolves the POI label to a 3D position using the object mask + depth image.
3. Computes an approach orientation — for pushes/pulls, the surface normal of the local mask region ([_calculate_surface_normal](cognitive_bt_framework/src/skills/skill_handler.py#L1002)); for side grasps, a perpendicular approach to the object pose.
4. Emits an `ExecutableAction` with concrete 3D pose + parameters ([skill_handler.py:13-23](cognitive_bt_framework/src/skills/skill_handler.py#L13-L23)).

The resulting `InstantiatedSkill.action_sequence` is then handed to `DirectSkillExecutor`, which drives CuRobo to plan and the xArm to execute.

---

## Repository layout

```
cognitive_bt_framework/
├── src/
│   ├── task_planner.py                       # entry point: NL task → skill commands
│   ├── skills/
│   │   ├── skill_generator.py                # LLM-based skill synthesis & retrieval
│   │   ├── skill_handler.py                  # symbolic skill → ExecutableAction sequence
│   │   ├── skill_executor.py                 # DirectSkillExecutor (camera+perception+exec)
│   │   ├── skill_executor_streamlined.py     # leaner executor variant
│   │   ├── primitive_parser.py               # regex parser for primitive vocabulary
│   │   └── run_skill_executor_streamlined.py # CLI entry point
│   ├── vision/
│   │   ├── perception_system.py              # orchestrator: detection → POIs → surfaces
│   │   ├── interaction_point_detector_v2.py  # primary geometric POI detector
│   │   ├── object_detection/yolo.py          # HybridYOLO: YOLO-World + FastSAM
│   │   ├── sam/                              # SAM2 / FastSAM mask generators
│   │   ├── realsense.py / zed_camera.py      # RGB-D camera backends
│   │   └── mapping/                          # voxel map + SLAM
│   ├── robot_interface/
│   │   ├── xarm_curobo_interface.py          # CuRobo motion planning, ROS-free
│   │   └── xarm_curobo_interface_streamlined.py
│   ├── llm_interface/
│   │   ├── llm_interface_openai.py           # OpenAI (default o3)
│   │   ├── llm_interface_claude.py
│   │   └── llm_query_logger.py               # shared LLM call/response logger
│   ├── data_recorder.py                      # per-execution session logging
│   └── sim/                                  # AI2-THOR & robosuite simulators
├── testing/                                  # standalone component tests
└── utils/
```

---

## Setup

Copy the env template and fill in absolute paths for your machine:

```bash
cp .env.example .env
# edit .env to point PROJECT_ROOT, CUROBO_XARM7_URDF, SAM2_CHECKPOINT, etc.
```

Required env vars:

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT` | Root of this repo |
| `COGNITIVE_BT_DATA_DIR` | Where data-collection runs are written |
| `CUROBO_XARM7_URDF` | xArm7 URDF inside the CuRobo install |
| `GROUNDINGDINO_CFG` / `GROUNDINGDINO_CHKPT` | Optional GroundingDINO assets |
| `SAM2_CHECKPOINT` | SAM 2 model weights |
| `LLM_CONVO_HTML_OUTPUT` | Optional path for LLM conversation export |

Install Python dependencies:

```bash
pip install -r requirements.txt
pip install -r trajectory_optimization_requirements.txt   # CuRobo extras
```

You also need: CuRobo, FastSAM-x weights, a YOLO-World checkpoint, and SAM 2 weights.

---

## Running a task

```python
from cognitive_bt_framework.src.task_planner import TaskPlanner

planner = TaskPlanner(
    robot_ip="192.168.1.224",
    use_llm=True,
    enable_llm_logging=True,     # writes every LLM query/response to disk
    enable_data_recording=False,
)

skill_sequence = planner.analyze_task("open the bottle on the table")
# → ["detect_object(bottle)", "open(bottle)", ...]

for skill in skill_sequence:
    planner.skill_executor.execute(skill)
```

For headless replay of previously generated plans, see [replay_successful_plans.py](replay_successful_plans.py).

---

## LLM call surface

All LLM traffic flows through `LLMInterfaceOpenAI`. Three call sites:

| Caller | Method | Purpose |
|---|---|---|
| `TaskPlanner._llm_task_decomposition` | `get_response_with_image` | NL task → skill commands |
| `SkillGenerator.{generate,find_similar,adapt}_skill` | `query_llm_sync` | Skill synthesis & retrieval |
| Behavior-tree paths in `cbtf.py` | `query_llm` (async) | Legacy BT generation |

`LLMQueryLogger` is instantiated by `TaskPlanner` and threaded through every downstream component, so a single run produces one consolidated log with image data stripped.

---

## Data collection

`DataRecorder` ([data_recorder.py](cognitive_bt_framework/src/data_recorder.py)) captures per-session:
- The natural-language task and image context
- Every skill command issued and its instantiation
- Joint trajectories at a configurable timestep
- Final success/failure outcome

Aggregate analysis: [analyze_success_rates.py](analyze_success_rates.py).

---

## Notes

- The robot interface is **ROS-free**. CuRobo is driven directly via Python.
- RealSense is the default camera; ZED is supported via [zed_camera.py](cognitive_bt_framework/src/vision/zed_camera.py).
- Generated skills persist to `stored_skills/` (or whatever `skills_dir` is passed to `SkillGenerator`); each skill is a JSON next to the prompt images it was generated from, so the LLM-VLM trace is fully reconstructable.
