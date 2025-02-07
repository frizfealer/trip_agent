import asyncio
import os

import gradio as gr
from dotenv import load_dotenv

from agent.chat_openai_factory import ChatOpenAIFactory
from agent.google_place_api import GooglePlaceAPI
from agent.trip_agent import TripAgent
from agent.trip_preference import TripPreference


def create_trip_planner():
    load_dotenv()

    async def get_categories_fn(city: str):
        chat_factory = ChatOpenAIFactory(openai_api_key=os.getenv("OPENAI_API_KEY"))
        google_api = GooglePlaceAPI(api_key=os.getenv("GOOGLE_PLACE_API_KEY"))
        agent = TripAgent(chat_factory, google_api)
        categories = await agent.get_categories(city)
        return ", ".join(categories)

    async def get_trip_recommendations(
        city: str,
        n_recommendations: int,
        people_count: int,
        budget: int,
        interests: str,
        progress=gr.Progress(),
    ):
        chat_factory = ChatOpenAIFactory(openai_api_key=os.getenv("OPENAI_API_KEY"))
        google_api = GooglePlaceAPI(api_key=os.getenv("GOOGLE_PLACE_API_KEY"))
        agent = TripAgent(chat_factory, google_api)

        preferences = TripPreference(
            location=city,
            people_count=people_count,
            budget=budget,
            interests=interests.split(","),
            trip_days=7,
        )

        recommendations = await agent.get_recommendations(
            n_recommendations, preferences
        )
        progress(0.5, desc=f"Found {len(recommendations)} recommendations")

        output = ""
        total_recs = len(recommendations)
        for i, rec in enumerate(recommendations, 1):
            progress(
                (0.5 + (i / total_recs) * 0.5),
                desc=f"Formatting recommendation {i}/{total_recs}",
            )
            output += f"""
 # 🏛️ {i}. {rec.name} 
 **📝 Agenda**: {rec.agenda} <br>
 **🎯 Category**: {rec.category} <br>
 **💰 Estimated Cost**: ${rec.cost} <br>
 **⏰ Duration**: {rec.duration}h <br>
 **👥 Group Size**: {rec.peopleCount} <br>
 **📅 Best Time**: {rec.time} <br>
 **⭐ Rating**: {rec.rating}/5 <br>
{f'**🌐 Website**: {rec.website_uri} <br>' if rec.website_uri != 'NA' else ''}
{f'**📄 Editorial Summary**: {rec.editorial_summary} <br>' if rec.editorial_summary != 'NA' else ''}
 {f'**🕒 Hours**: {rec.regular_opening_hours} <br>' if rec.regular_opening_hours != 'NA' else ''}
 **📸 Photos**:<br>
<div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: start;">
{"".join(f'<div style="flex: 0 0 200px;"><img src="{photo}" alt="Photo of {rec.name}" style="width: 100%; height: 150px; object-fit: cover;"></div>' for photo in rec.photos) if rec.photos else ""}
</div>

 ---
"""
        return output

    async def get_itinerary(
        recommendations_str: str,
        city: str,
        trip_days: int,
        people_count: int,
        budget: int,
        progress=gr.Progress(),
    ):
        chat_factory = ChatOpenAIFactory(openai_api_key=os.getenv("OPENAI_API_KEY"))
        google_api = GooglePlaceAPI(api_key=os.getenv("GOOGLE_PLACE_API_KEY"))
        agent = TripAgent(chat_factory, google_api)

        preferences = TripPreference(
            location=city,
            trip_days=trip_days,
            people_count=people_count,
            budget=budget,
            interests=["all"],
        )
        progress(0, desc="Generating itinerary...")
        output = await agent.get_itinerary_with_reflection(
            recommendations=recommendations_str,
            trip_preference=preferences,
            reflection_num=2,
        )
        return output

    with gr.Blocks(title="Trip Planner") as demo:
        gr.Markdown("# 🌍 Trip Planner")

        with gr.Tab("Categories"):
            city_input1 = gr.Textbox(label="City", placeholder="e.g., Tokyo")
            categories_button = gr.Button("Get Categories")
            categories_output = gr.Textbox(label="Interesting Categories")

        with gr.Tab("Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    city_input2 = gr.Textbox(label="City", placeholder="e.g., Tokyo")
                    n_recommendations = gr.Number(
                        label="Number of Recommendations", value=3, minimum=1
                    )
                    people_count = gr.Number(label="Group Size", value=2, minimum=1)
                    budget = gr.Number(label="Budget (USD)", value=1000)
                    interests = gr.Textbox(
                        label="Interests (comma-separated)",
                        placeholder="e.g., food, culture, history",
                    )
                    rec_button = gr.Button("Get Recommendations")
                with gr.Column(scale=2):
                    recommendations_output = gr.Markdown(label="Recommendations")

        with gr.Tab("Itinerary Generator"):
            with gr.Row():
                with gr.Column(scale=1):
                    city_input3 = gr.Textbox(label="City", placeholder="e.g., Tokyo")
                    trip_days = gr.Number(
                        label="Trip Duration (days)", value=3, minimum=1
                    )
                    people_count2 = gr.Number(label="Group Size", value=2, minimum=1)
                    budget2 = gr.Number(label="Budget (USD)", value=1000)
                    recommendations_input = gr.Textbox(
                        label="Paste Recommendations", lines=10
                    )
                    itinerary_button = gr.Button("Generate Itinerary")
                with gr.Column(scale=2):
                    itinerary_output = gr.Textbox(label="Itinerary")

        categories_button.click(
            fn=lambda x: asyncio.run(get_categories_fn(x)),
            inputs=city_input1,
            outputs=categories_output,
        )

        rec_button.click(
            fn=lambda city, n, people, budget, interests: asyncio.run(
                get_trip_recommendations(city, n, people, budget, interests)
            ),
            inputs=[city_input2, n_recommendations, people_count, budget, interests],
            outputs=recommendations_output,
        )

        itinerary_button.click(
            fn=lambda rec_str, city, days, people, budget: asyncio.run(
                get_itinerary(rec_str, city, days, people, budget)
            ),
            inputs=[
                recommendations_input,
                city_input3,
                trip_days,
                people_count2,
                budget2,
            ],
            outputs=itinerary_output,
        )

    return demo


if __name__ == "__main__":
    demo = create_trip_planner()
    demo.launch(share=True)
