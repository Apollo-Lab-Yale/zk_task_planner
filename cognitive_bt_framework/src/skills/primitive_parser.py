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
    
    # Regex patterns for different primitive types including point-based patterns
    PATTERNS = {
        # Original patterns (keeping for backward compatibility)
        'push_surface': re.compile(
            r'push\('
            r'\[(?P<surface_keywords>[^\]]+)\]'
            r'(?:,\s*(?:is_parallel_surface=)?(?P<is_parallel_surface>true|false))?'
            r'(?:,\s*(?:is_button=)?(?P<is_button>true|false))?'
            r'(?:,\s*(?:has_pivot=)?(?P<has_pivot>true|false))?'
            r'(?:,\s*(?:hinge_location=)?(?:\'|")?(?P<hinge_location>top|bottom|left|right|)(?:\'|")?)?'
            r'(?:,\s*(?:pivot_point=)?(?P<pivot_point>[^)]+))?'  # Keep for backward compatibility
            r'\)'
        ),
        'pull_surface': re.compile(
            r'pull\('
            r'\[(?P<surface_keywords>[^\]]+)\]'
            r'(?:,\s*(?:is_parallel_surface=)?(?P<is_parallel_surface>true|false))?'
            r'(?:,\s*(?:is_button=)?(?P<is_button>true|false))?'
            r'(?:,\s*(?:has_pivot=)?(?P<has_pivot>true|false))?'
            r'(?:,\s*(?:hinge_location=)?(?:\'|")?(?P<hinge_location>top|bottom|left|right|)(?:\'|")?)?'
            r'(?:,\s*(?:pivot_point=)?(?P<pivot_point>[^)]+))?'  # Keep for backward compatibility
            r'\)'
        ),
        'move_gripper_to_pose_keywords': re.compile(
            r'move_gripper_to_pose\('
            r'\[(?P<keywords>[^\]]+)\]'
            r'(?:,\s*(?:is_top_down_grasp=)?(?P<is_top_down_grasp>true|false))?'
            r'(?:,\s*(?:is_side_grasp=)?(?P<is_side_grasp>true|false))?'
            r'\)'
        ),
        
        # Point-based patterns with positional parameters
        'push_positional': re.compile(
            r'push\('
            r'(?:\'|")(?P<point_label>[A-Za-z0-9]+)(?:\'|")'  # Point label (e.g., 'A')
            r'(?:,\s*(?:\'|")(?P<force_direction>parallel|perpendicular)(?:\'|"))?'  # Force direction
            r'(?:,\s*(?P<is_button>true|false))?'  # Is button
            r'(?:,\s*(?P<has_pivot>true|false))?'  # Has pivot
            r'(?:,\s*(?:\'|")(?P<hinge_location>top|bottom|left|right|)(?:\'|"))?'  # Hinge location (can be empty)
            r'\)'
        ),
        'pull_positional': re.compile(
            r'pull\('
            r'(?:\'|")(?P<point_label>[A-Za-z0-9]+)(?:\'|")'  # Point label (e.g., 'A')
            r'(?:,\s*(?:\'|")(?P<force_direction>parallel|perpendicular)(?:\'|"))?'  # Force direction
            r'(?:,\s*(?P<is_button>true|false))?'  # Is button
            r'(?:,\s*(?P<has_pivot>true|false))?'  # Has pivot
            r'(?:,\s*(?:\'|")(?P<hinge_location>top|bottom|left|right|)(?:\'|"))?'  # Hinge location (can be empty)
            r'\)'
        ),
        'move_gripper_to_pose_positional': re.compile(
            r'move_gripper_to_pose\('
            r'(?:\'|")(?P<point_label>[A-Za-z0-9]+)(?:\'|")'  # Point label (e.g., 'A')
            r'(?:,\s*(?P<is_top_down_grasp>true|false))?'  # Is top-down grasp
            r'(?:,\s*(?P<is_side_grasp>true|false))?'  # Is side grasp
            r'\)'
        ),
        
        # Point-based patterns with named parameters - FIXED PATTERN HERE
        'push_named': re.compile(
            r'push\('
            r'point_label=(?:\'|")(?P<point_label>[A-Za-z0-9]+)(?:\'|")'  # Point label (e.g., 'A')
            r'(?:,\s*force_direction=(?:\'|")(?P<force_direction>parallel|perpendicular)(?:\'|"))?'  # Force direction
            r'(?:,\s*is_button=(?P<is_button>true|false))?'  # Is button
            r'(?:,\s*has_pivot=(?P<has_pivot>true|false))?'  # Has pivot
            r'(?:,\s*(?:hinge_location=(?:\'|")(?P<hinge_location>top|bottom|left|right|)(?:\'|")|pivot_point_label=(?:\'|")(?P<pivot_point_label>[A-Za-z0-9]+)(?:\'|")))?'  # Hinge location or pivot point label
            r'\)'
        ),
        'pull_named': re.compile(
            r'pull\('
            r'point_label=(?:\'|")?(?P<point_label>[A-Za-z0-9]+)(?:\'|")?'  # Optional quotes
            r'(?:,\s*force_direction=(?:\'|")?(?P<force_direction>parallel|perpendicular)(?:\'|")?)?'
            r'(?:,\s*is_button=(?P<is_button>true|false))?'
            r'(?:,\s*has_pivot=(?P<has_pivot>true|false))?'
            r'(?:,\s*(?:hinge_location=(?:\'|")?(?P<hinge_location>top|bottom|left|right|)(?:\'|")?|pivot_point_label=(?:\'|")?(?P<pivot_point_label>[A-Za-z0-9]+)(?:\'|")?))?'  # Hinge location or pivot point label
            r'\)'
        ),
        'move_gripper_to_pose_named': re.compile(
            r'move_gripper_to_pose\(\s*'
            r'point_label=(?:\'|")(?P<point_label>[A-Za-z0-9]+)(?:\'|")\s*'
            r'(?:,\s*is_top_down_grasp=(?P<is_top_down_grasp>true|false))?\s*'
            r'(?:,\s*is_side_grasp=(?P<is_side_grasp>true|false))?\s*'
            r'\)'
        ),
        
        # Simple primitives (unchanged)
        'close_gripper': re.compile(
            r'close_gripper\(\)'
        ),
        'open_gripper': re.compile(
            r'open_gripper\(\)'
        ),
        'retract_gripper': re.compile(
            r'retract_gripper\(\)'
        ),
        
        # NEW: Twist primitive with direction parameter
        'twist': re.compile(
            r'twist\('
            r'(?:\'|")(?P<direction>clockwise|counterclockwise)(?:\'|")'
            r'\)'
        ),
        
        # Additional pattern for the exact failing case
        'push_exact_fail_case': re.compile(
            r'push\('
            r'point_label=(?:\'|")(?P<point_label>[A-Za-z0-9]+)(?:\'|")'
            r',\s*force_direction=(?:\'|")(?P<force_direction>parallel|perpendicular)(?:\'|")'
            r',\s*is_button=(?P<is_button>true|false)'
            r',\s*has_pivot=(?P<has_pivot>true|false)'
            r',\s*hinge_location=(?:\'|")(?P<hinge_location>top|bottom|left|right|)(?:\'|")'
            r'\)'
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
        """
        # Clean the input string
        primitive_str = primitive_str.strip()
        
        # Remove leading dash or bullet point if present (common in YAML lists)
        if primitive_str.startswith('-') or primitive_str.startswith('*'):
            primitive_str = primitive_str[1:].strip()
        
        # Try each pattern
        for pattern_name, pattern in cls.PATTERNS.items():
            match = pattern.match(primitive_str)
            if match:
                # Extract the base action type from the pattern name
                action_type = pattern_name.split('_')[0]
                return cls._create_parsed_primitive(action_type, match, primitive_str, pattern_name)
        
        print(f"Warning: Could not parse primitive string: {primitive_str}")
        return None

    @classmethod
    def _create_parsed_primitive(cls, action_type: str, match: re.Match, raw_string: str, pattern_name: str) -> ParsedPrimitive:
        """Create ParsedPrimitive object from regex match"""
        parameters = {}
        
        # Fix the action_type extraction to handle multi-part action names
        if pattern_name.startswith('move_gripper_to_pose'):
            action_type = 'move_gripper_to_pose'
        elif pattern_name.startswith('push'):
            action_type = 'push'
        elif pattern_name.startswith('pull'):
            action_type = 'pull'
        elif pattern_name.startswith('close_gripper'):
            action_type = 'close_gripper'
        elif pattern_name.startswith('open_gripper'):
            action_type = 'open_gripper'
        elif pattern_name.startswith('retract_gripper'):
            action_type = 'retract_gripper'
        elif pattern_name.startswith('twist'):
            action_type = 'twist'
        
        # Get all named groups from the match
        match_dict = match.groupdict()
        
        # Process based on pattern type
        if pattern_name in ['push_surface', 'pull_surface', 'move_gripper_to_pose_keywords']:
            # Process for old-style patterns
            # Process keywords if present
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
            for param in ['is_parallel_surface', 'is_button', 'has_pivot', 
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
            
            # Process hinge_location for surface patterns
            if 'hinge_location' in match_dict and match_dict['hinge_location'] is not None:
                parameters['hinge_location'] = match_dict['hinge_location']
        elif pattern_name == 'twist':
            # Process twist primitive
            if 'direction' in match_dict and match_dict['direction'] is not None:
                parameters['direction'] = match_dict['direction']
        else:
            # Process for point-based patterns (both positional and named)
            if 'point_label' in match_dict and match_dict['point_label'] is not None:
                parameters['point_label'] = match_dict['point_label']
            
            if 'force_direction' in match_dict and match_dict['force_direction'] is not None:
                parameters['force_direction'] = match_dict['force_direction']
            
            if 'hinge_location' in match_dict and match_dict['hinge_location'] is not None:
                # Empty string is a valid value for when no hinge is specified
                parameters['hinge_location'] = match_dict['hinge_location']
            elif 'pivot_point_label' in match_dict and match_dict['pivot_point_label'] is not None:
                # Backward compatibility: convert old pivot_point_label to hinge_location
                parameters['pivot_point_label'] = match_dict['pivot_point_label']
            
            # Process boolean parameters
            for param in ['is_button', 'has_pivot', 'is_top_down_grasp', 'is_side_grasp']:
                if param in match_dict and match_dict[param] is not None:
                    parameters[param] = match_dict[param].lower() == 'true'
            
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
    def format_primitive(action_type: str, use_named_params: bool = False, **kwargs) -> str:
        """
        Format a primitive action string with given parameters
        
        Args:
            action_type: Type of primitive action
            use_named_params: Whether to use named parameters (param=value) format
            **kwargs: Parameters for the primitive
            
        Returns:
            Formatted primitive string
        """
        if action_type in ['close_gripper', 'open_gripper', 'retract_gripper']:
            return f"{action_type}()"
        
        # Handle twist primitive
        if action_type == 'twist':
            direction = kwargs.get('direction', 'clockwise')
            return f"twist('{direction}')"
        
        # Determine if using point-based or surface-based format
        if 'point_label' in kwargs:
            # Point-based format
            if action_type in ['push', 'pull']:
                if use_named_params:
                    # Named parameters format
                    result = f"{action_type}(point_label='{kwargs['point_label']}'"
                    
                    # Add optional parameters with names
                    if 'force_direction' in kwargs:
                        result += f", force_direction='{kwargs['force_direction']}'"
                    
                    for param in ['is_button', 'has_pivot']:
                        if param in kwargs:
                            result += f", {param}={str(kwargs[param]).lower()}"
                    
                    if 'hinge_location' in kwargs:
                        result += f", hinge_location='{kwargs['hinge_location']}'"
                    elif 'pivot_point_label' in kwargs:
                        # Backward compatibility
                        result += f", pivot_point_label='{kwargs['pivot_point_label']}'"
                else:
                    # Positional parameters format
                    result = f"{action_type}('{kwargs['point_label']}'"
                    
                    # Add optional parameters
                    if 'force_direction' in kwargs:
                        result += f", '{kwargs['force_direction']}'"
                    
                    for param in ['is_button', 'has_pivot']:
                        if param in kwargs:
                            result += f", {str(kwargs[param]).lower()}"
                    
                    if 'hinge_location' in kwargs:
                        result += f", '{kwargs['hinge_location']}'"
                    elif 'pivot_point_label' in kwargs:
                        # Backward compatibility
                        result += f", '{kwargs['pivot_point_label']}'"
                
                result += ')'
                return result
                
            elif action_type == 'move_gripper_to_pose':
                if use_named_params:
                    # Named parameters format
                    result = f"{action_type}(point_label='{kwargs['point_label']}'"
                    
                    for param in ['is_top_down_grasp', 'is_side_grasp']:
                        if param in kwargs:
                            result += f", {param}={str(kwargs[param]).lower()}"
                else:
                    # Positional parameters format
                    result = f"{action_type}('{kwargs['point_label']}'"
                    
                    for param in ['is_top_down_grasp', 'is_side_grasp']:
                        if param in kwargs:
                            result += f", {str(kwargs[param]).lower()}"
                
                result += ')'
                return result
        else:
            # Original surface-based format
            if action_type in ['push', 'pull']:
                # Format surface keywords array
                surface_keywords_str = f"[{', '.join(f'{k}' for k in kwargs.get('surface_keywords', []))}]"
                result = f"{action_type}({surface_keywords_str}"
                
                # Add optional parameters with names
                optional_params = ['is_parallel_surface', 'is_button', 'has_pivot', 'hinge_location', 'pivot_point']
                for param in optional_params:
                    if param in kwargs:
                        # Add comma
                        result += ', '
                        
                        # Add the parameter name and value
                        if param in ['is_parallel_surface', 'is_button', 'has_pivot']:
                            result += f"{param}={str(kwargs[param]).lower()}"
                        elif param == 'hinge_location':
                            result += f"hinge_location='{kwargs[param]}'"
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