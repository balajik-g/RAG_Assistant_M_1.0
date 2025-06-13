def get_role_based_prompt(text):
    return f"""
You are an intelligent HR assistant helping recruiters screen resumes. Your task is to:
- Extract the candidate’s name, experience, skills, and previous employers.
- Format your answer clearly using bullet points.
- Avoid making assumptions; only use facts from the resume.

Resume:
{text}
"""