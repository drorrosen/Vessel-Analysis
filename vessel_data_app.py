
import streamlit as st

st.set_page_config(
    page_title="Real-Time Data Science Dashboards",
    layout="wide",
    initial_sidebar_state="expanded",)

def intro():
    import streamlit as st

    st.write("# Welcome to Shipping Data Analysis project! 👋")
    st.sidebar.success("Select a Dashboard above.")

    st.markdown(
        """
        Data analysis Project: This project contains the analysis of the files
        """
    )




def dashboard_1():
    import streamlit as st
    import pandas as pd
    import plotly.express as px





    #Title
    st.title('historical tonnage counts USG Dashboard')
    st.write('')
    # Load the dataset
    @st.cache_data
    def load_data():
        df = pd.read_csv('historical tonnage counts USG.csv',parse_dates=['Date'], index_col='Date')
        df['Vessel Count Difference'] = df['Vessel Count'].diff()
        return df

    df = load_data()

    df['Vessel DWT Difference'] = df['Vessel DWT'].diff()
    df['Vessel Count Difference'] = df['Vessel Count'].diff()



    st.subheader('Descriptive Statistics')

    # create three columns
    kpi1, kpi2, kpi3 = st.columns(3)

    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Vessel Count Average 🚢",
        value=round(df['Vessel Count'].mean())
    )

    kpi2.metric(
        label="Vessel Count Median 🚢",
        value=round(df['Vessel Count'].median())
    )

    kpi3.metric(
        label="Vessel Count SD 🚢",
        value=f" {round(df['Vessel Count'].std(),2)} ")




    # create three columns
    kpi11, kpi12, kpi13 = st.columns(3)

    # fill in those three columns with respective metrics or KPIs
    kpi11.metric(
        label="Vessel DWT Average ⚓",
        value=round(df['Vessel DWT'].mean())
    )

    kpi12.metric(
        label="Vessel DWT Median ⚓",
        value=round(df['Vessel DWT'].median())
    )

    kpi13.metric(
        label="Vessel DWT SD ⚓",
        value=f" {round(df['Vessel DWT'].std(),2)} ")


    st.divider()
    st.subheader('Data Visualization')

    # create two columns for charts
    fig_col1, fig_col2 = st.columns(2)



    with fig_col1:
        # Plot the histogram for Deadweight Tonnage (DWT)
        fig5 = px.histogram(df, x='Vessel DWT', title='Distribution of Deadweight Tonnage (DWT)', nbins=50)
        fig5.update_xaxes(title_text='Deadweight Tonnage (DWT)')
        fig5.update_yaxes(title_text='Frequency')
        #fig5.update_layout(width=1200, height=1200)

        st.write(fig5)

    with fig_col2:
        # Plot the histogram for Vessel Count
        fig6 = px.histogram(df, x='Vessel Count', title='Distribution of Vessel Count', nbins=50)
        fig6.update_xaxes(title_text='Vessel Count')
        fig6.update_yaxes(title_text='Frequency')
        #fig6.update_layout(width=1200, height=1200)

        st.write(fig6)



    # create two columns for charts
    fig_col21, fig_col22 = st.columns(2)



    with fig_col21:
        # Plot Vessel Count Over Time using Plotly Express
        fig1 = px.line(df, x=df.index, y='Vessel Count', title='Vessel Count Over Time')
        fig1.update_xaxes(title_text='Date')
        fig1.update_yaxes(title_text='Vessel Count')
        fig1.update_layout(width=1200, height=1200)

        st.write(fig1)


    with fig_col22:
        # Plot Vessel Count Difference Over Time using Plotly Express
        fig2 = px.line(df, x=df.index, y='Vessel Count Difference', title='Vessel Count Difference Over Time', color_discrete_sequence=['red'])
        fig2.update_xaxes(title_text='Date')
        fig2.update_yaxes(title_text='Vessel Count Difference')
        fig2.update_layout(width=1200, height=1200)

        st.write(fig2)


    # create two columns for charts
    fig_col31, fig_col32 = st.columns(2)



    with fig_col31:
        # Plot DWT Over Time using Plotly Express
        fig3 = px.line(df, x=df.index, y='Vessel DWT', title='Deadweight Tonnage (DWT) Over Time')
        fig3.update_xaxes(title_text='Date')
        fig3.update_yaxes(title_text='Deadweight Tonnage (DWT)')
        fig3.update_layout(width=1200, height=1200)

        st.write(fig3)



    with fig_col32:
        # Plot Vessel Count Difference Over Time using Plotly Express
        fig4 = px.line(df, x=df.index, y='Vessel DWT Difference', title='Vessel DWT Difference Over Time', color_discrete_sequence=['red'])
        fig4.update_xaxes(title_text='Date')
        fig4.update_yaxes(title_text='Vessel DWT Difference')
        #fig4.update_layout(width=1200, height=1200)

        st.write(fig4)


    st.divider()
    st.subheader("Calculating the correlation between DWT and Vessel Count")


    # Plot Correlation Between DWT and Vessel Count Over Time
    correlation_value = df['Vessel Count'].corr(df['Vessel DWT'])
    fig5 = px.scatter(df, x='Vessel Count', y='Vessel DWT', title=f'Correlation Between DWT and Vessel Count Over Time (Correlation: {correlation_value:.2f})')
    fig5.update_xaxes(title_text='Vessel Count')
    fig5.update_yaxes(title_text='Deadweight Tonnage (DWT)')
    #fig5.update_layout(width=1200, height=1200)

    st.write(fig5)


    st.divider()
    st.subheader('Detailed Data View')
    st.dataframe(df)





def dashboard_2():
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # File paths
    s4a_58_file = 's4a_58.xlsx'
    baltic_exchange_data_file = 'BALTIC EXCHANGE DATA s1c58 2022.xlsx'

    # Step 1: Data Loading
    try:
        s4a_58_df = pd.read_excel(s4a_58_file, parse_dates=['Date'])
        s4a_58_df.drop_duplicates(keep='first', inplace=True)

        baltic_exchange_data_df = pd.read_excel(baltic_exchange_data_file,parse_dates=['Date'])
        baltic_exchange_data_df.drop_duplicates(keep='first', inplace=True)

    except Exception as e:
        print(f"An error occurred while loading the datasets: {e}")


    combined_df = pd.concat([s4a_58_df, baltic_exchange_data_df])


    #title
    st.title('Shipping Route Analytics Dashboard')

    #user input for description
    selected_description = st.selectbox("Select Description/Location", combined_df['Description'].unique())

    #Filter data based on user selection
    filtered_data = combined_df[combined_df['Description'] == selected_description]

    #display statistics
    st.subheader('Descriptive Statistics')
    st.dataframe(filtered_data.describe(include='all'))

    st.divider()
    st.subheader('Time series Plot')
    # Display plot using Plotly Express
    fig = px.line(filtered_data, x='Date', y='Value', title='Value in $ Over Time', color_discrete_sequence=['red'])
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Value ($)')
    #fig.update_layout(width=1200, height=1200)
    st.write(fig)


    st.divider()
    st.subheader('Detailed Data View')
    st.dataframe(filtered_data)




def dashboard_3():
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    from scipy import stats
    import numpy as np

    #read daily file
    @st.cache_data
    def load_data():
        df = pd.read_csv('daily_panama.csv')
        return df
    df = load_data()
    df['DIRECTION (N/S)'] = df['DIRECTION (N/S)'].str.lower()
    df.rename(columns={'DIRECTION (N/S)': 'DIRECTION'}, inplace=True)

    #--- SIDEBAR ---
    st.sidebar.header('Please Filter Here:')
    year = st.sidebar.multiselect('Select Year',
                                  options=df['YEAR'].unique(),
                                  default=df['YEAR'].unique())

    direction = st.sidebar.multiselect('Select Direction',
                                        options=df['DIRECTION'].unique(),
                                       default=df['DIRECTION'].unique())


    df_selection = df.query(
        "DIRECTION == @direction & YEAR == @year"
    )

    # ---- MAINPAGE ----
    st.title(":PANAMA Dashboard: Analysis")
    st.markdown("##")
    st.dataframe(df_selection)

    df_selection.drop(['DIRECTION','YEAR'], axis=1)

    # Function to calculate IQR
    def calculate_iqr(series):
        Q3 = series.quantile(0.75)
        Q1 = series.quantile(0.25)
        IQR = Q3 - Q1
        return IQR

    # Calculate the IQR for each 'TRANSIT DATE'
    iqr_series = df.groupby('TRANSIT DATE')['WAITING TIME'].apply(calculate_iqr).reset_index()
    iqr_series.rename(columns={'WAITING TIME': 'IQR'}, inplace=True)



    agg_df_selection = df_selection.groupby(['TRANSIT DATE']).agg({
            'WAITING TIME': 'median',
            'BEAM (pies)': 'median',
            'DRAF (Pies)': 'median',
            'transit_booking_days': 'median',
            'TRANSIT DATE': 'count'

    }).rename(columns={'TRANSIT DATE': 'SHIP_COUNT'}).reset_index()


    agg_df_selection['iqr_waiting_time'] = iqr_series['IQR']

    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(agg_df_selection['WAITING TIME'].dropna()))

    # Get boolean array indicating the presence of outliers
    outliers = (z_scores > 3)

    # Indices of outliers
    outlier_indices = np.where(outliers)[0]

    # Values of outliers
    outlier_values = agg_df_selection['WAITING TIME'].dropna().iloc[outlier_indices]

    # Removing the outliers from the 'WAITING TIME' column
    agg_df_selection = agg_df_selection.drop(outlier_values.index)






    st.subheader('Time series Plots')
    # create two columns for charts
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        # Display plot using Plotly Express
        fig = px.line(agg_df_selection, x='TRANSIT DATE', y='WAITING TIME', title='TRANSIT DATE VS WAITING TIME', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Transit Date')
        fig.update_yaxes(title_text='Median Waiting Time (days)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    with fig_col2:
        # Display plot using Plotly Express
        fig = px.line(agg_df_selection, x='TRANSIT DATE', y='BEAM (pies)', title='TRANSIT DATE VS BEAM (pies)', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Transit Date')
        fig.update_yaxes(title_text='Median Beam (pies)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    # create two columns for charts
    fig_col11, fig_col12 = st.columns(2)

    with fig_col11:
        # Display plot using Plotly Express
        fig = px.line(agg_df_selection, x='TRANSIT DATE', y='DRAF (Pies)', title='TRANSIT DATE VS DRAF (Pies)', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Transit Date')
        fig.update_yaxes(title_text='Median DRAF (Pies)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    with fig_col12:
        # Display plot using Plotly Express
        fig = px.line(agg_df_selection, x='TRANSIT DATE', y='transit_booking_days', title='TRANSIT DATE VS transit_booking_days', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Transit Date')
        fig.update_yaxes(title_text='Median transit_booking_days')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)


    # create two columns for charts
    fig_col41, fig_col42 = st.columns(2)

    with fig_col41:
        # Display plot using Plotly Express
        fig = px.line(agg_df_selection, x='TRANSIT DATE', y='SHIP_COUNT', title='TRANSIT DATE VS ship_count', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Transit Date')
        fig.update_yaxes(title_text='Median SHIP_COUNT')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    with fig_col42:
        # Display plot using Plotly Express
        fig = px.line(agg_df_selection, x='TRANSIT DATE', y='iqr_waiting_time', title='TRANSIT DATE VS iqr_waiting_time', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Transit Date')
        fig.update_yaxes(title_text='Median iqr_waiting_time')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    st.divider()

    st.subheader('Correlation Heatmap between variables')
    import plotly.graph_objects as go

    # Calculate the correlation matrix
    correlation_matrix = agg_df_selection[['DRAF (Pies)', 'BEAM (pies)', 'WAITING TIME', 'transit_booking_days', 'SHIP_COUNT']].corr()

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        hoverongaps = False,
        colorscale='YlOrRd'))

    # Customize the layout
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Variables')
    )

    # Show the plot
    fig.update_layout(width=1200, height=1200)

    st.write(fig)

    st.divider()
    st.subheader('Seasonal Patterns related to congestion')
    # Convert 'TRANSIT DATE' to datetime format
    agg_df_selection['TRANSIT DATE'] = pd.to_datetime(agg_df_selection['TRANSIT DATE'])

    # Extract month and year from 'TRANSIT DATE'
    agg_df_selection['MONTH'] = agg_df_selection['TRANSIT DATE'].dt.month
    # Extract day of the week from 'TRANSIT DATE'
    # Monday is 0 and Sunday is 6
    agg_df_selection['DAY_OF_WEEK'] = agg_df_selection['TRANSIT DATE'].dt.dayofweek

    # Extract the quarter and store it in a new column 'QUARTER'
    agg_df_selection['QUARTER'] = agg_df_selection['TRANSIT DATE'].dt.to_period("Q")
    agg_df_selection['QUARTER'] = agg_df_selection['QUARTER'].astype(str)

    # Group data by month and year, then calculate the median waiting time for each group
    grouped_df_waiting_time = agg_df_selection.groupby(['MONTH'])['WAITING TIME'].median().reset_index()
    grouped_df_ship_count = agg_df_selection.groupby(['MONTH'])['SHIP_COUNT'].median().reset_index()

    # create two columns for charts
    fig_col51, fig_col52 = st.columns(2)

    with fig_col51:
        # Display plot using Plotly Express
        fig = px.line(grouped_df_waiting_time, x='MONTH', y='WAITING TIME', title='Monthly Seasonal Patterns in Canal Congestion (Median Waiting Time)', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Month')
        fig.update_yaxes(title_text='Median Waiting Time (months)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    with fig_col52:
        # Display plot using Plotly Express
        fig = px.line(grouped_df_ship_count, x='MONTH', y='SHIP_COUNT', title='Monthly Seasonal Patterns in Canal Congestion (Median Ship Count)', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Month')
        fig.update_yaxes(title_text='Median Ship Count')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    # Group data by dayofweek, then calculate the median waiting time for each group
    grouped_df_waiting_time_dowk = agg_df_selection.groupby(['DAY_OF_WEEK'])['WAITING TIME'].median().reset_index()
    grouped_df_ship_count_dowk = agg_df_selection.groupby(['DAY_OF_WEEK'])['SHIP_COUNT'].median().reset_index()



    # create two columns for charts
    fig_col61, fig_col62 = st.columns(2)

    with fig_col61:
        # Display plot using Plotly Express
        fig = px.line(grouped_df_waiting_time_dowk, x='DAY_OF_WEEK', y='WAITING TIME', title='Weekly Seasonal Patterns in Canal Congestion (Median Waiting Time)', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Day of Week')
        fig.update_yaxes(title_text='Median Waiting Time (days)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    with fig_col62:
        # Display plot using Plotly Express
        fig = px.line(grouped_df_ship_count_dowk, x='DAY_OF_WEEK', y='SHIP_COUNT', title='Weekly Seasonal Patterns in Canal Congestion (Median Ship Count)', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Day of Week')
        fig.update_yaxes(title_text='Median Ship Count')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    # Group data by quarter, then calculate the median waiting time for each group
    grouped_df_waiting_time_quarter = agg_df_selection.groupby(['QUARTER'])['WAITING TIME'].median().reset_index()
    grouped_df_ship_count_quarter = agg_df_selection.groupby(['QUARTER'])['SHIP_COUNT'].median().reset_index()

    # create two columns for charts
    fig_col71, fig_col72 = st.columns(2)

    with fig_col71:
        # Display plot using Plotly Express
        fig = px.line(grouped_df_waiting_time_quarter, x='QUARTER', y='WAITING TIME', title='Quarterly Seasonal Patterns in Canal Congestion (Median Waiting Time)', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Quarter')
        fig.update_yaxes(title_text='Median Waiting Time (Quarter)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    with fig_col72:
        # Display plot using Plotly Express
        fig = px.line(grouped_df_ship_count_quarter, x='QUARTER', y='SHIP_COUNT', title='Quarterly Seasonal Patterns in Canal Congestion (Median Ship Count)', color_discrete_sequence=['red'])
        fig.update_xaxes(title_text='Quarter')
        fig.update_yaxes(title_text='Median Ship Count')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)



    st.divider()
    st.subheader('Comparison between directions')

    # Group data by direction, then calculate the median waiting time for each group
    grouped_df_waiting_time_direction = df_selection.groupby(['TRANSIT DATE','DIRECTION'])['WAITING TIME'].median().reset_index()
    grouped_df_ship_count_direction = df_selection.groupby(['TRANSIT DATE','DIRECTION'])['YEAR'].count().reset_index()
    grouped_df_beam_direction = df_selection.groupby(['TRANSIT DATE','DIRECTION'])['BEAM (pies)'].median().reset_index()
    grouped_df_draf_direction = df_selection.groupby(['TRANSIT DATE','DIRECTION'])['DRAF (Pies)'].median().reset_index()


    # create two columns for charts
    fig_col81, fig_col82 = st.columns(2)

    with fig_col81:
        # Display plot using Plotly Express
        fig = px.line(grouped_df_waiting_time_direction, x='TRANSIT DATE', y='WAITING TIME', color='DIRECTION', title='Directional Patterns in Canal Congestion (Median Waiting Time)', color_discrete_sequence=['orange', 'blue'])
        fig.update_xaxes(title_text='Direction')
        fig.update_yaxes(title_text='Median Waiting Time (days)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    with fig_col82:
        # Display plot using Plotly Express
        fig = px.line(grouped_df_ship_count_direction, x='TRANSIT DATE', y='YEAR', color='DIRECTION', title='Directional Patterns in Canal Congestion (Ship Count)', color_discrete_sequence=['orange', 'blue'])
        fig.update_xaxes(title_text='Direction')
        fig.update_yaxes(title_text='Median Ship Count')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    #create two columns for charts
    fig_col91, fig_col92 = st.columns(2)

    #plotting comparisons between directions for beam and draf
    with fig_col91:
        fig = px.line(grouped_df_beam_direction, x='TRANSIT DATE', y='BEAM (pies)', color='DIRECTION', title='Directional Patterns in Median Beam (pies)', color_discrete_sequence=['orange', 'blue'])
        fig.update_xaxes(title_text='Direction')
        fig.update_yaxes(title_text='Median Beam (pies)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)

    with fig_col92:
        fig = px.line(grouped_df_draf_direction, x='TRANSIT DATE', y='DRAF (Pies)', color='DIRECTION', title='Directional Patterns in Median Draf (pies)', color_discrete_sequence=['orange', 'blue'])
        fig.update_xaxes(title_text='Direction')
        fig.update_yaxes(title_text='Median Draf (pies)')
        #fig.update_layout(width=1200, height=1200)
        st.write(fig)


    #Correlations between directions - all variables
    st.divider()
    st.subheader('Correlations between directions')

    #corr between median waiting times of southbound and northbound, using the file grouped_df_waiting_time_direction
    pivot_df = grouped_df_waiting_time_direction.pivot(index='TRANSIT DATE', columns='DIRECTION', values='WAITING TIME').reset_index()
    correlation_between_directions = pivot_df['northbound'].corr(pivot_df['southbound'], method='spearman')
    st.write('''Correlation between northbound and southbound waiting times: ''',round(correlation_between_directions,2))

    #corr between median ship counts of southbound and northbound, using the file grouped_df_ship_count_direction
    pivot_df = grouped_df_ship_count_direction.pivot(index='TRANSIT DATE', columns='DIRECTION', values='YEAR').reset_index()
    correlation_between_directions = pivot_df['northbound'].corr(pivot_df['southbound'], method='spearman')
    st.write('''Correlation between northbound and southbound ship counts: ''',round(correlation_between_directions,2))

    #corr between median beam of southbound and northbound, using the file grouped_df_beam_direction
    pivot_df = grouped_df_beam_direction.pivot(index='TRANSIT DATE', columns='DIRECTION', values='BEAM (pies)').reset_index()
    correlation_between_directions = pivot_df['northbound'].corr(pivot_df['southbound'], method='spearman')
    st.write('''Correlation between northbound and southbound beam: ''',round(correlation_between_directions,2))

    #corr between median draf of southbound and northbound, using the file grouped_df_draf_direction
    pivot_df = grouped_df_draf_direction.pivot(index='TRANSIT DATE', columns='DIRECTION', values='DRAF (Pies)').reset_index()
    correlation_between_directions = pivot_df['northbound'].corr(pivot_df['southbound'], method='spearman')
    st.write('''Correlation between northbound and southbound draf: ''',round(correlation_between_directions,2))


    #southbound dataset to download - cannal_congestion medians for southbound -
    # Group data by direction southbound, then calculate the median waiting time for each group
    df_southbound = df_selection[df_selection['DIRECTION'] == 'southbound']
    grouped_df_waiting_time_direction_southbound = df_southbound.groupby(['TRANSIT DATE'])['WAITING TIME'].median().reset_index()
    grouped_df_ship_count_direction_southbound = df_southbound.groupby(['TRANSIT DATE'])['YEAR'].count().reset_index()
    grouped_df_beam_direction_southbound = df_southbound.groupby(['TRANSIT DATE'])['BEAM (pies)'].median().reset_index()
    grouped_df_draf_direction_southbound = df_southbound.groupby(['TRANSIT DATE'])['DRAF (Pies)'].median().reset_index()

    #merge these series to one dataset
    merged_df_southbound = pd.merge(grouped_df_waiting_time_direction_southbound, grouped_df_ship_count_direction_southbound, on='TRANSIT DATE')
    merged_df_southbound = pd.merge(merged_df_southbound, grouped_df_beam_direction_southbound, on='TRANSIT DATE')
    merged_df_southbound = pd.merge(merged_df_southbound, grouped_df_draf_direction_southbound, on='TRANSIT DATE')
    merged_df_southbound.rename(columns={'WAITING TIME': 'WAITING TIME SOUTHBOUND', 'YEAR': 'SHIP COUNT SOUTHBOUND', 'BEAM (pies)': 'BEAM SOUTHBOUND', 'DRAF (Pies)': 'DRAF SOUTHBOUND'}, inplace=True)



    #northbound dataset to download - cannal_congestion medians for northbound -
    # Group data by direction northbound, then calculate the median waiting time for each group
    df_northbound = df_selection[df_selection['DIRECTION'] == 'northbound']
    grouped_df_waiting_time_direction_northbound = df_northbound.groupby(['TRANSIT DATE'])['WAITING TIME'].median().reset_index()
    grouped_df_ship_count_direction_northbound = df_northbound.groupby(['TRANSIT DATE'])['YEAR'].count().reset_index()
    grouped_df_beam_direction_northbound = df_northbound.groupby(['TRANSIT DATE'])['BEAM (pies)'].median().reset_index()
    grouped_df_draf_direction_northbound = df_northbound.groupby(['TRANSIT DATE'])['DRAF (Pies)'].median().reset_index()

    #merge these series to one dataset
    merged_df_northbound = pd.merge(grouped_df_waiting_time_direction_northbound, grouped_df_ship_count_direction_northbound, on='TRANSIT DATE')
    merged_df_northbound = pd.merge(merged_df_northbound, grouped_df_beam_direction_northbound, on='TRANSIT DATE')
    merged_df_northbound = pd.merge(merged_df_northbound, grouped_df_draf_direction_northbound, on='TRANSIT DATE')
    merged_df_northbound.rename(columns={'WAITING TIME': 'WAITING TIME NORTHBOUND', 'YEAR': 'SHIP COUNT NORTHBOUND', 'BEAM (pies)': 'BEAM NORTHBOUND', 'DRAF (Pies)': 'DRAF NORTHBOUND'}, inplace=True)

    #concatenate the two datasets
    merged_df = pd.merge(merged_df_southbound, merged_df_northbound, on='TRANSIT DATE')

    merged_df.to_csv('merged_df_direction_panama.csv')






















page_names_to_funcs = {
    "Introduction": intro,
    "historical tonnage counts USG Dashboard": dashboard_1,
    "Shipping Route Analytics Dashboard": dashboard_2,
    "Panama Analysis": dashboard_3

}

dashboard_name = st.sidebar.selectbox("Choose a dashboard", page_names_to_funcs.keys())
page_names_to_funcs[dashboard_name]()
#%%


