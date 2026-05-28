#!/usr/bin/env python3
"""
Simplified Skill Executor Runner

Simple script to execute single manipulation skills.
Usage: python run_skill_executor_streamlined.py <skill> <object>

Examples:
    python run_skill_executor_streamlined.py open bottle
    python run_skill_executor_streamlined.py pickup cup
    python run_skill_executor_streamlined.py place cup table
"""

import sys
import argparse

# Import the streamlined skill executor
from cognitive_bt_framework.src.skills.skill_executor_streamlined import DirectSkillExecutor


def execute_skill(skill_name: str, target_object: str, destination: str = None, robot_ip: str = "192.168.1.224"):
    """
    Execute a single manipulation skill

    Args:
        skill_name: Name of the skill (e.g., 'open', 'pickup', 'place')
        target_object: Target object name (e.g., 'bottle', 'cup')
        destination: Destination for place skill (optional)
        robot_ip: Robot IP address

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"  Executing: {skill_name} {target_object}")
    if destination:
        print(f"  Destination: {destination}")
    print(f"{'='*60}\n")

    try:
        # Initialize executor
        print("Initializing skill executor...")
        executor = DirectSkillExecutor(
            robot_ip=robot_ip,
            camera_params={'width': 640, 'height': 480, 'fps': 30},
            fast_mode=True
        )
        print("✅ Executor initialized\n")

        # Build skill sequence (always detect object first)
        skill_sequence = [("detect_object", target_object)]

        # Add the manipulation skill
        if skill_name.lower() == "place":
            if not destination:
                print("❌ Error: 'place' skill requires a destination")
                return False
            # Detect destination object first
            skill_sequence.append(("detect_object", destination))
            skill_sequence.append((skill_name, f"{target_object},{destination}"))
        else:
            skill_sequence.append((skill_name, target_object))

        # Execute the sequence
        print(f"Executing skill sequence:")
        for i, (skill, params) in enumerate(skill_sequence, 1):
            print(f"  {i}. {skill} {params}")
        print()

        success, message = executor.execute_skill_sequence(
            skill_sequence,
            task_context=f"{skill_name} {target_object}"
        )

        # Print result
        print(f"\n{'='*60}")
        if success:
            print(f"  ✅ SUCCESS: {skill_name} {target_object}")
            print(f"  {message}")
        else:
            print(f"  ❌ FAILED: {skill_name} {target_object}")
            print(f"  Error: {message}")
        print(f"{'='*60}\n")

        # Cleanup
        executor.shutdown()

        return success

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Execute a single manipulation skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s open bottle
  %(prog)s pickup cup
  %(prog)s place cup table
  %(prog)s close drawer
  %(prog)s switchon light

Available skills:
  detect_object  - Detect and locate an object
  open          - Open doors, drawers, containers
  close         - Close doors, drawers, containers
  pickup        - Pick up an object
  place         - Place object at location (requires destination)
  switchon      - Turn on switches/buttons
  switchoff     - Turn off switches/buttons
  twist         - Twist/rotate objects
        """
    )

    parser.add_argument(
        "skill",
        type=str,
        help="Skill to execute (e.g., 'open', 'pickup', 'place')"
    )

    parser.add_argument(
        "object",
        type=str,
        help="Target object name (e.g., 'bottle', 'cup')"
    )

    parser.add_argument(
        "destination",
        type=str,
        nargs='?',
        default=None,
        help="Destination for 'place' skill (optional)"
    )

    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.1.224",
        help="Robot IP address (default: 192.168.1.224)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute skill
    success = execute_skill(
        skill_name=args.skill,
        target_object=args.object,
        destination=args.destination,
        robot_ip=args.robot_ip
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
