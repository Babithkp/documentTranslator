import re


class RowParser:
    def parse_course_row(self, text):
        """
        Parse rows like:
        FCOS 204 ADULT DEVELOPMENT 2 A 8.00
        """

        text = " ".join(text.split())

        pattern = (
            r'^'
            r'([A-Z]{2,6}\s*\d{2,4})\s+'   # code
            r'(.+?)\s+'                    # course name
            r'(\d+)\s+'                    # credits
            r'([A-F][+]?)\s+'             # grade
            r'(\d+(?:\.\d+)?)'            # GPA
            r'$'
        )

        match = re.match(pattern, text)

        if not match:
            return None

        return {
            "code": match.group(1),
            "course": match.group(2),
            "credits": match.group(3),
            "grade": match.group(4),
            "gpa": match.group(5),
        }