import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ParsedPrimitive:
    """Data class to store parsed primitive information"""
    action_type: str
    parameters: Dict[str, Any]
    raw_string: str

class PrimitiveParser:
    """Parser for primitive action strings using regular expressions"""
    
    # Regex patterns for different primitive types
    PATTERNS = {
        'push': re.compile(
            r'push\('
            r'(?P<distance>\d*\.?\d+)'
            r'\)'
        ),
        'pull': re.compile(
            r'pull\('
            r'(?P<distance>\d*\.?\d+)'
            r'\)'
        ),
        'moveGripperToPose': re.compile(
            r'moveGripperToPose\('
            r'\[(?P<keywords>[^\]]+)\]'
            r'(?:,\s*(?P<is_grasp>true|false))?'
            r'\)'
        ),
        'close_gripper': re.compile(
            r'close_gripper\(\)'
        ),
        'release': re.compile(
            r'release\(\)'
        ),
        'retractGripper': re.compile(
            r'retractGripper\(\)'
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
            >>> primitive = parser.parse_primitive("push(0.1)")
            >>> print(primitive.action_type)  # 'push'
            >>> print(primitive.parameters)   # {'distance': 0.1}
        """
        # Clean the input string
        primitive_str = primitive_str.strip()
        
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
        
        # Process keywords if present
        if 'keywords' in match_dict:
            keywords_str = match_dict['keywords']
            # Handle both quoted and unquoted keywords
            keywords = []
            for k in keywords_str.split(','):
                k = k.strip()
                # Remove quotes if present
                if (k.startswith("'") and k.endswith("'")) or (k.startswith('"') and k.endswith('"')):
                    k = k[1:-1]
                keywords.append(k)
            parameters['keywords'] = keywords
            
        # Process distance for push/pull
        if 'distance' in match_dict:
            parameters['distance'] = float(match_dict['distance'])
            
        # Process is_grasp for moveGripperToPose
        if 'is_grasp' in match_dict and match_dict['is_grasp'] is not None:
            parameters['is_grasp'] = match_dict['is_grasp'].lower() == 'true'
            
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
            >>> PrimitiveParser.format_primitive('push', distance=0.1)
            "push(0.1)"
            >>> PrimitiveParser.format_primitive('moveGripperToPose', keywords=['top', 'handle'], is_grasp=True)
            "moveGripperToPose(['top', 'handle'], true)"
        """
        if action_type in ['close_gripper', 'release', 'retractGripper']:
            return f"{action_type}()"
            
        if action_type in ['push', 'pull']:
            return f"{action_type}({kwargs['distance']})"
            
        if action_type == 'moveGripperToPose':
            base = f"{action_type}([{', '.join(kwargs['keywords'])}])"
            if 'is_grasp' in kwargs:
                base = base[:-1] + f", is_grasp={str(kwargs['is_grasp']).lower()})"
            return base
            
        return f"{action_type}()"

def test_primitive_parser():
    """Test the PrimitiveParser with various primitive strings"""
    parser = PrimitiveParser()
    
    # Test cases
    test_primitives = [
        "push(0.1)",
        "pull(0.05)",
        "moveGripperToPose([top, handle], is_grasp=true)",
        "moveGripperToPose([approach_point])",
        "close_gripper()",
        "release()",
        "retractGripper()"
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
        ('push', {'distance': 0.1}),
        ('pull', {'distance': 0.05}),
        ('moveGripperToPose', {'keywords': ['top', 'handle'], 'is_grasp': True}),
        ('moveGripperToPose', {'keywords': ['approach_point']}),
        ('close_gripper', {}),
        ('release', {}),
        ('retractGripper', {})
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