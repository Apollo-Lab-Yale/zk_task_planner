import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ParsedPrimitive:
    """Data class to store parsed primitive information with action type and parameters"""
    action_type: str
    parameters: Dict[str, Any]
    raw_string: str

class PrimitiveParser:
    """Parser for primitive action strings using regular expressions"""
    
    # Regex patterns for different primitive types
    PATTERNS = {
        'push': re.compile(
            r'push\('
            r'\[(?P<surface_keywords>[^\]]+)\]'
            r'(?:,\s*(?:is_parallel_surface=)?(?P<is_parallel_surface>true|false))?'
            r'(?:,\s*(?:is_bottom=)?(?P<is_bottom>true|false))?'
            r'(?:,\s*(?:has_pivot=)?(?P<has_pivot>true|false))?'
            r'(?:,\s*(?:pivot_point=)?(?P<pivot_point>[^)]+))?'
            r'\)'
        ),
        'pull': re.compile(
            r'pull\('
            r'\[(?P<surface_keywords>[^\]]+)\]'
            r'(?:,\s*(?:is_parallel_surface=)?(?P<is_parallel_surface>true|false))?'
            r'(?:,\s*(?:is_bottom=)?(?P<is_bottom>true|false))?'
            r'(?:,\s*(?:has_pivot=)?(?P<has_pivot>true|false))?'
            r'(?:,\s*(?:pivot_point=)?(?P<pivot_point>[^)]+))?'
            r'\)'
        ),
        'move_gripper_to_pose': re.compile(
            r'move_gripper_to_pose\('
            r'\[(?P<keywords>[^\]]+)\]'
            r'(?:,\s*(?:is_top_down_grasp=)?(?P<is_top_down_grasp>true|false))?'
            r'(?:,\s*(?:is_side_grasp=)?(?P<is_side_grasp>true|false))?'
            r'\)'
        ),
        'close_gripper': re.compile(
            r'close_gripper\(\)'
        ),
        'open_gripper': re.compile(
            r'open_gripper\(\)'
        ),
        'retract_gripper': re.compile(
            r'retract_gripper\(\)'
        )
    }

    @classmethod
    def parse_primitive(cls, primitive_str: str) -> Optional[ParsedPrimitive]:
        """
        Parse a primitive action string into its components
        
        Args:
            primitive_str: String containing the primitive action
            
        Returns:
            ParsedPrimitive object if successful, None if parsing fails
            
        Examples:
            >>> parser = PrimitiveParser()
            >>> primitive = parser.parse_primitive("push([table_surface], is_parallel_surface=true, is_bottom=false)")
            >>> print(primitive.action_type)  # 'push'
            >>> print(primitive.parameters)   # {'surface_keywords': ['table_surface'], ...}
        """
        # Clean the input string
        primitive_str = primitive_str.strip()
        
        # Remove leading dash or bullet point if present (common in YAML lists)
        if primitive_str.startswith('-') or primitive_str.startswith('*'):
            primitive_str = primitive_str[1:].strip()
        
        # Try each pattern
        for action_type, pattern in cls.PATTERNS.items():
            match = pattern.match(primitive_str)
            if match:
                return cls._create_parsed_primitive(action_type, match, primitive_str)
        
        print(f"Warning: Could not parse primitive string: {primitive_str}")
        return None

    @classmethod
    def _create_parsed_primitive(cls, action_type: str, match: re.Match, raw_string: str) -> ParsedPrimitive:
        """Create ParsedPrimitive object from regex match"""
        parameters = {}
        
        # Get all named groups from the match
        match_dict = match.groupdict()
        
        # Process keywords if present (for move_gripper_to_pose or surface_keywords)
        for keyword_param in ['keywords', 'surface_keywords']:
            if keyword_param in match_dict and match_dict[keyword_param] is not None:
                keywords_str = match_dict[keyword_param]
                # Handle both quoted and unquoted keywords
                keywords = []
                for k in keywords_str.split(','):
                    k = k.strip()
                    # Remove quotes if present
                    if (k.startswith("'") and k.endswith("'")) or (k.startswith('"') and k.endswith('"')):
                        k = k[1:-1]
                    keywords.append(k)
                parameters[keyword_param] = keywords
        
        # Process boolean parameters
        for param in ['is_parallel_surface', 'is_bottom', 'has_pivot', 
                      'is_top_down_grasp', 'is_side_grasp']:
            if param in match_dict and match_dict[param] is not None:
                parameters[param] = match_dict[param].lower() == 'true'
        
        # Process string parameters
        for param in ['surface', 'pivot_point']:
            if param in match_dict and match_dict[param] is not None:
                parameters[param] = match_dict[param].strip()
                # Remove quotes if present
                if ((parameters[param].startswith("'") and parameters[param].endswith("'")) or 
                    (parameters[param].startswith('"') and parameters[param].endswith('"'))):
                    parameters[param] = parameters[param][1:-1]
            
        return ParsedPrimitive(
            action_type=action_type,
            parameters=parameters,
            raw_string=raw_string
        )

    @classmethod
    def validate_primitive_sequence(cls, primitives: List[str]) -> List[ParsedPrimitive]:
        """
        Validate and parse a sequence of primitive actions
        
        Args:
            primitives: List of primitive action strings
            
        Returns:
            List of successfully parsed primitives
            
        Raises:
            ValueError: If any primitive fails to parse
        """
        parsed_primitives = []
        failed_primitives = []
        
        for primitive_str in primitives:
            parsed = cls.parse_primitive(primitive_str)
            if parsed:
                parsed_primitives.append(parsed)
            else:
                failed_primitives.append(primitive_str)
        
        if failed_primitives:
            raise ValueError(
                f"Failed to parse primitives:\n" +
                "\n".join(f"- {p}" for p in failed_primitives)
            )
            
        return parsed_primitives

    @staticmethod
    def format_primitive(action_type: str, **kwargs) -> str:
        """
        Format a primitive action string with given parameters
        
        Args:
            action_type: Type of primitive action
            **kwargs: Parameters for the primitive
            
        Returns:
            Formatted primitive string
            
        Examples:
            >>> PrimitiveParser.format_primitive('push', surface_keywords=['table_surface'], is_parallel_surface=True)
            "push([table_surface], is_parallel_surface=true)"
            >>> PrimitiveParser.format_primitive('move_gripper_to_pose', keywords=['handle', 'knob'], is_top_down_grasp=False, is_side_grasp=True)
            "move_gripper_to_pose(['handle', 'knob'], is_top_down_grasp=false, is_side_grasp=true)"
        """
        if action_type in ['close_gripper', 'open_gripper', 'retract_gripper']:
            return f"{action_type}()"
            
        if action_type in ['push', 'pull']:
            # Format surface keywords array
            surface_keywords_str = f"[{', '.join(f'{k}' for k in kwargs.get('surface_keywords', []))}]"
            result = f"{action_type}({surface_keywords_str}"
            
            # Add optional parameters with names
            optional_params = ['is_parallel_surface', 'is_bottom', 'has_pivot', 'pivot_point']
            for param in optional_params:
                if param in kwargs:
                    # Add comma
                    result += ', '
                    
                    # Add the parameter name and value
                    if param in ['is_parallel_surface', 'is_bottom', 'has_pivot']:
                        result += f"{param}={str(kwargs[param]).lower()}"
                    else:
                        result += f"{param}={str(kwargs[param])}"
            
            # Ensure the string ends with a closing parenthesis
            result += ')'
                
            return result
            
        if action_type == 'move_gripper_to_pose':
            # Format keywords array
            keywords_str = f"[{', '.join(f'{k}' for k in kwargs.get('keywords', []))}]"
            result = f"{action_type}({keywords_str}"
            
            # Add optional parameters with names
            if 'is_top_down_grasp' in kwargs:
                result += f", is_top_down_grasp={str(kwargs['is_top_down_grasp']).lower()}"
                
                # Add is_side_grasp if is_top_down_grasp is present
                if 'is_side_grasp' in kwargs:
                    result += f", is_side_grasp={str(kwargs['is_side_grasp']).lower()}"
            elif 'is_side_grasp' in kwargs:
                # If only is_side_grasp is present, add is_top_down_grasp first
                result += f", is_top_down_grasp=false, is_side_grasp={str(kwargs['is_side_grasp']).lower()}"
                
            result += ")"
            return result
            
        return f"{action_type}()"

def test_primitive_parser():
    """Test the PrimitiveParser with various primitive strings"""
    parser = PrimitiveParser()
    
    # Test cases with both named and unnamed parameters
    test_primitives = [
        "push([table, surface], is_parallel_surface=true, is_bottom=false, has_pivot=false)",
        "push([table, surface], true, false, false)",
        "pull([drawer, handle], is_parallel_surface=false, is_bottom=true, has_pivot=true, pivot_point=top_edge)",
        "pull([drawer, handle], false, true, true, top_edge)",
        "move_gripper_to_pose([handle, knob, grip], is_top_down_grasp=true, is_side_grasp=false)",
        "move_gripper_to_pose([handle, knob, grip], true, false)",
        "move_gripper_to_pose([toggle, switch])",
        "- move_gripper_to_pose(['duck', 'toy'], is_top_down_grasp=true, is_side_grasp=false)",
        "- move_gripper_to_pose(['destination'], is_top_down_grasp=false, is_side_grasp=false)",
        "close_gripper()",
        "open_gripper()",
        "retract_gripper()"
    ]
    
    print("Testing primitive parser...")
    for primitive_str in test_primitives:
        parsed = parser.parse_primitive(primitive_str)
        if parsed:
            print(f"\nParsed: {primitive_str}")
            print(f"Action Type: {parsed.action_type}")
            print(f"Parameters: {parsed.parameters}")
        else:
            print(f"\nFailed to parse: {primitive_str}")
            
    # Test primitive sequence validation
    print("\nTesting sequence validation...")
    try:
        parsed_sequence = parser.validate_primitive_sequence(test_primitives)
        print(f"Successfully parsed {len(parsed_sequence)} primitives")
    except ValueError as e:
        print(f"Validation failed: {e}")
        
    # Test primitive formatting
    print("\nTesting primitive formatting...")
    test_cases = [
        ('push', {'surface_keywords': ['table', 'surface'], 'is_parallel_surface': True, 'is_bottom': False}),
        ('pull', {'surface_keywords': ['drawer', 'handle'], 'is_parallel_surface': False, 'is_bottom': True, 'has_pivot': True, 'pivot_point': 'top_edge'}),
        ('move_gripper_to_pose', {'keywords': ['handle', 'knob', 'grip'], 'is_top_down_grasp': True, 'is_side_grasp': False}),
        ('move_gripper_to_pose', {'keywords': ['toggle', 'switch']}),
        ('close_gripper', {}),
        ('open_gripper', {}),
        ('retract_gripper', {})
    ]
    
    for action_type, params in test_cases:
        formatted = parser.format_primitive(action_type, **params)
        print(f"\n{action_type}:")
        print(f"  Formatted: {formatted}")
        # Verify we can parse what we format
        parsed = parser.parse_primitive(formatted)
        if parsed:
            print("  Successfully parsed formatted string")
        else:
            print("  Failed to parse formatted string")

if __name__ == "__main__":
    test_primitive_parser()