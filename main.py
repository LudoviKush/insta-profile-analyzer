import instaloader
import os
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import cv2
from PIL import Image
import io
from dotenv import load_dotenv
import ffmpeg
import numpy as np
from datetime import datetime
import json
import subprocess
import time
load_dotenv()  # Load environment variables from .env file

# Configure Vertex AI
vertexai.init(project="tr-media-analysis", location="europe-central2")

# Generation config and safety settings
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

class InstagramAnalyzer:
    def __init__(self):
        self.loader = instaloader.Instaloader()
        self.temp_dir = Path("temp_downloads")
        self.output_dir = Path("output")
        # Create both directories if they don't exist
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.model = GenerativeModel("gemini-1.5-pro-002")
    
    def login(self, username, password):
        try:
            print(f"Attempting login method for {username}...")
            time.sleep(2)
            
            # Configure custom settings
            self.loader.context.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            self.loader.context.max_connection_attempts = 3
            
            # Try to login without session handling first
            self.loader.context.login(username, password)
            print("Login successful!")
            return True
                
        except Exception as e:
            print(f"Login process failed: {str(e)}")
            print("Please try:")
            print("1. Verify credentials manually on Instagram")
            print("2. Check for account locks or security checks")
            print("3. Wait a few minutes before trying again")
            return False

    def analyze_profile(self, profile_username):
        try:
            time.sleep(3)
            
            try:
                self.loader.context.login(
                    os.getenv('INSTAGRAM_USERNAME'),
                    os.getenv('INSTAGRAM_PASSWORD')
                )
            except Exception as login_error:
                print("Refreshing session...")
                time.sleep(5)
            
            profile = instaloader.Profile.from_username(self.loader.context, profile_username)
            time.sleep(2)
            
            profile_data = {
                "username": profile.username,
                "full_name": profile.full_name,
                "biography": profile.biography,
                "external_url": profile.external_url,
                "followers": profile.followers,
                "following": profile.followees,
                "posts_count": profile.mediacount,
                "is_business": profile.is_business_account,
                "business_category": profile.business_category_name
            }

            # Analyze latest 3 posts (both images and videos)
            posts_data = []
            post_count = 0
            
            for post in profile.get_posts():
                if post_count >= 3:
                    break
                    
                # Download the post
                self.loader.download_post(post, target=self.temp_dir)
                
                # Try to safely get post attributes
                try:
                    hashtags = list(post.caption_hashtags) if hasattr(post, 'caption_hashtags') else []
                    mentioned_users = list(post.tagged_users) if hasattr(post, 'tagged_users') else []
                    location = post.location if hasattr(post, 'location') else None
                    
                    post_data = {
                        "post_url": f"https://www.instagram.com/p/{post.shortcode}",
                        "caption": post.caption,
                        "likes": post.likes,
                        "comments": post.comments,
                        "timestamp": post.date_utc.isoformat(),
                        "is_video": post.is_video,
                        "location": location,
                        "hashtags": hashtags,
                        "mentioned_users": mentioned_users,
                    }

                    # Find all downloaded files for this post
                    post_files = list(self.temp_dir.glob(f"{post.date_utc.strftime('%Y-%m-%d_%H-%M-%S')}_UTC*"))
                    
                    if post.is_video:
                        # Handle video posts
                        video_files = [f for f in post_files if f.suffix.lower() == '.mp4']
                        if video_files:
                            video_path = video_files[0]
                            post_data["media_analysis"] = self.analyze_video_with_gemini(video_path)
                            post_data["media_type"] = "video"
                            post_data["video_view_count"] = post.video_view_count if hasattr(post, 'video_view_count') else None
                    else:
                        # Handle image posts (including carousels)
                        image_files = [f for f in post_files if f.suffix.lower() in ['.jpg', '.jpeg']]
                        if image_files:
                            image_analyses = []
                            for image_path in image_files:
                                analysis = self.analyze_image_with_gemini(image_path)
                                image_analyses.append(analysis)
                            post_data["media_analysis"] = image_analyses
                            post_data["media_type"] = "carousel" if len(image_files) > 1 else "image"
                    
                    posts_data.append(post_data)
                    post_count += 1
                    
                except Exception as post_error:
                    print(f"Error processing post: {str(post_error)}")
                    continue

            # Analyze profile with Gemini
            profile_analysis = self.analyze_with_gemini(profile_data, posts_data)
            
            # Clean up temporary files
            self.cleanup()
            
            return {
                "profile_data": profile_data,
                "posts_data": posts_data,
                "overall_analysis": profile_analysis
            }
            
        except Exception as e:
            print(f"Error analyzing profile: {str(e)}")
            self.cleanup()
            return None

    def analyze_video_with_gemini(self, video_path):
        try:
            # Get video information using ffprobe
            probe_cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]

            probe_output = subprocess.check_output(probe_cmd).decode('utf-8')
            video_info = json.loads(probe_output)

            # Extract duration from video info - fixed property access
            duration = float(video_info['format']['duration'])
            frames = []

            # Calculate timestamps for beginning, middle, and end
            timestamps = [0, duration/2, duration-1]

            for t in timestamps:
                try:
                    # Create output filename for this frame
                    output_frame = self.temp_dir / f"frame_{t}.jpg"
                    
                    # Simplified ffmpeg command
                    cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-ss', str(t),
                        '-vframes', '1',
                        '-f', 'image2',
                        '-y',
                        str(output_frame)
                    ]

                    # Execute ffmpeg command with timeout
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode != 0:
                        print(f"FFmpeg error at {t}s: {result.stderr}")
                        continue

                    if output_frame.exists():
                        with open(output_frame, 'rb') as img_file:
                            image_bytes = img_file.read()
                            frames.append(
                                Part.from_data(
                                    data=image_bytes,
                                    mime_type="image/jpeg"
                                )
                            )
                        # Clean up frame
                        output_frame.unlink()

                except subprocess.TimeoutExpired:
                    print(f"Frame extraction timed out at {t}s")
                    continue
                except Exception as e:
                    print(f"Frame extraction error at {t}s: {str(e)}")
                    continue

            if not frames:
                return "Video analysis failed: Could not extract any frames"

            # Create prompt
            prompt = Part.from_text("""
            Analyze these frames from the video content and provide insights on:
            1. Content quality and production value
            2. Marketing effectiveness
            3. Authenticity indicators
            4. Potential red flags for scams
            5. Target audience engagement
            Please focus on visual elements, setting, and overall presentation.
            """)

            # Combine prompt and frames
            parts = [prompt] + frames

            try:
                response = self.model.generate_content(
                    parts,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                return response.text
            except Exception as api_error:
                print(f"API Error: {str(api_error)}")
                return "Video analysis failed due to API error."

        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            return f"Video analysis failed: {str(e)}"

    def analyze_with_gemini(self, profile_data, posts_data):
        prompt = f"""
Analyze this Instagram profile focusing on these key metrics and provide actionable recommendations:

1. Visual Branding (30% weight)
- Brand Style Guide Assessment:
  * Color palette consistency and psychology
  * Typography hierarchy and readability
  * Logo placement and variations
  * Design element standardization
  * Image filters and editing style
- Visual Content Quality:
  * Image resolution and composition
  * Text overlay legibility
  * Negative space utilization
  * Grid layout harmony

2. Content Strategy (30% weight)
- Content Format Distribution:
  * Ratio of images/videos/carousels
  * Story highlights organization
  * Reels implementation
  * Live session frequency
- Content Pillars:
  * Industry expertise demonstration
  * Behind-the-scenes content
  * Employee spotlights
  * Client success stories
  * Educational content depth
- Content Calendar Assessment:
  * Posting consistency
  * Content series identification
  * Seasonal relevance
  * Campaign integration

3. Engagement Optimization (20% weight)
- Community Building Tactics:
  * Question formats in captions
  * Poll and quiz implementation
  * User-generated content integration
  * Response time to comments
- Call-to-Action Effectiveness:
  * CTA variety and placement
  * Link-in-bio optimization
  * Story swipe-up usage (if available)
  * Direct message prompts

4. Technical Performance (20% weight)
- Profile Optimization:
  * Business account features utilization
  * Bio keyword optimization
  * Highlight cover consistency
  * Contact button implementation
- Growth Metrics:
  * Follower-to-following ratio
  * Engagement rate calculation
  * Reach vs. impressions
  * Hashtag performance analysis

Profile Information:
{json.dumps(profile_data, indent=2)}

Posts Analysis:
{json.dumps(posts_data, indent=2)}

Please provide:
1. Detailed scoring for each category (0-100)
2. Evidence-based analysis with specific post examples
3. Priority-ranked improvement actions:
   - Immediate actions (next 30 days)
   - Short-term improvements (60-90 days)
   - Long-term strategy (6 months)
4. Competitive benchmark against industry standards
5. Specific content ideas for next 30 days, including:
   - Post types and formats
   - Caption templates
   - Hashtag groups
   - Optimal posting times
6. Resource allocation recommendations:
   - Design tools needed
   - Content creation workflows
   - Team skill requirements
   - Automation opportunities

Format the response with clear sections, bullet points, and markdown formatting for readability.
Focus on quantifiable metrics and specific examples rather than general observations.
"""

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return response.text

    def cleanup(self):
        # Remove temporary downloaded files
        for file in self.temp_dir.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                print(f"Error deleting {file}: {str(e)}")

    def analyze_image_with_gemini(self, image_path):
        try:
            # Read image file in binary mode
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Create image Part directly from bytes
            image_part = Part.from_data(
                data=image_bytes,
                mime_type="image/jpeg"
            )
            
            # Create prompt as Part
            prompt = Part.from_text("""
            Analyze this image and provide insights on:
            1. Content quality and visual appeal
            2. Marketing effectiveness
            3. Authenticity indicators
            4. Professional vs. amateur presentation
            5. Target audience appeal
            6. Brand consistency
            7. Potential red flags or misleading elements
            
            Please focus on visual elements, composition, and overall presentation.
            """)

            # Combine prompt and image into parts list
            parts = [prompt, image_part]

            try:
                # Generate content
                response = self.model.generate_content(
                    parts,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                return response.text
            except Exception as api_error:
                print(f"Gemini API error: {str(api_error)}")
                return "Image analysis failed: API error"

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return f"Image analysis failed: {str(e)}"

def main():
    analyzer = InstagramAnalyzer()
    
    # Get credentials from environment variables
    username = os.getenv('INSTAGRAM_USERNAME')
    password = os.getenv('INSTAGRAM_PASSWORD')
    
    if not username or not password:
        print("Error: Instagram credentials not found in .env file")
        return
    
    if analyzer.login(username, password):
        target_profile = input("Enter the Instagram profile to analyze: ")
        results = analyzer.analyze_profile(target_profile)
        
        if results:
            # Save results to a file in the output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = analyzer.output_dir / f"analysis_{target_profile}_{timestamp}.json"
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            print(f"Analysis completed and saved to {output_file}")
            
            # Print summary
            print("\nProfile Analysis Summary:")
            print("-" * 50)
            print(results["overall_analysis"])
        else:
            print("Analysis failed.")
    else:
        print("Login failed. Please check your credentials in .env file.")

if __name__ == "__main__":
    main()
