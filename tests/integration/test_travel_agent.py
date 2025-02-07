import pytest


class TestIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, travel_agent, mock_llm_chain):
        """Test the entire recommendation flow"""
        recommendations = await travel_agent.get_recommendations(
            trip_days=5,
            people_count=2,
            locations="Taiwan",
            budget=2000,
            interests="Hiking, shopping, culture",
        )

        assert len(recommendations) > 0
        first_rec = recommendations[0]

        # Verify structure and constraints
        assert all(key in first_rec for key in MOCK_ATTRACTION.keys())
        assert travel_agent._is_duration_compatible(first_rec["duration"], 5)
        assert travel_agent._is_group_size_compatible(first_rec["peopleCount"], 2)
        assert travel_agent._is_within_budget(first_rec["budget"], 2000)
