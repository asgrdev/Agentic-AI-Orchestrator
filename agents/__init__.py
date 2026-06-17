"""
Agents Package

این پکیج شامل تمام agents و skills سیستم است.
"""

# بارگذاری خودکار advanced skills
try:
    from agents import advanced_skills
    print("✅ Advanced skills loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load advanced skills: {e}")

# بارگذاری خودکار specialized skills
try:
    from agents import specialized_skills
    print("✅ Specialized skills loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load specialized skills: {e}")

# بارگذاری خودکار document processing skills
try:
    from agents import document_processing_skills
    print("✅ Document processing skills loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load document processing skills: {e}")

# بارگذاری خودکار AI model skills
try:
    from agents import ai_model_skills
    print("✅ AI model skills loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load AI model skills: {e}")

# بارگذاری خودکار Sensor skills
try:
    from agents import sensor_skills
    print("✅ Sensor skills loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load sensor skills: {e}")

__all__ = [
    'advanced_skills',
    'specialized_skills',
    'document_processing_skills',
    'ai_model_skills',
    'sensor_skills',
]

