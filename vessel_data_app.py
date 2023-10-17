
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

    agg_df_selection = df_selection.groupby(['TRANSIT DATE']).agg({
        'WAITING TIME': 'median',
        'BEAM (pies)': 'median',
        'DRAF (Pies)': 'median',
        'transit_booking_days': 'median',

    }).reset_index()

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



    st.divider()

    st.subheader('Correlation Heatmap between variables')
    import plotly.graph_objects as go

    # Calculate the correlation matrix
    correlation_matrix = agg_df_selection[['DRAF (Pies)', 'BEAM (pies)', 'WAITING TIME', 'transit_booking_days']].corr()

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
    fig.update_layout(width=600, height=600)

    st.write(fig)




page_names_to_funcs = {
    "Introduction": intro,
    "historical tonnage counts USG Dashboard": dashboard_1,
    "Shipping Route Analytics Dashboard": dashboard_2,
    "Panama Analysis": dashboard_3

}

dashboard_name = st.sidebar.selectbox("Choose a dashboard", page_names_to_funcs.keys())
page_names_to_funcs[dashboard_name]()
#%%
