import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import numpy as np
import datetime
import pandas as pd
import json

# Inserting Color Labels
colorIncomeCensusgroup = pd.read_excel('https://raw.githubusercontent.com/Shai2u/RI_Microsimulator_July2021/main/assets/legacy_resources/colorINcomeCensusGroups.xlsx')  # New April 28
censusIncomeDict = dict(zip(colorIncomeCensusgroup['label'], colorIncomeCensusgroup['colors_']))
group_color_dict = pd.read_excel('https://raw.githubusercontent.com/Shai2u/RI_Microsimulator_July2021/main/assets/legacy_resources/group_color_jan_7.xlsx')  # read color dictionary
color_labels = pd.read_excel('https://raw.githubusercontent.com/Shai2u/RI_Microsimulator_July2021/main/assets/legacy_resources/jafan_7_color_labels.xlsx')
colorDict = dict(zip(color_labels['label'], color_labels['colors_']))
colorDictMerge = dict(
    zip(group_color_dict['group_name'], group_color_dict['colors_']))
geoJSONloc = "https://raw.githubusercontent.com/Shai2u/RI_Microsimulator_July2021/main/assets/legacy_resources/rib_feb_11.geojson"
csvResultsloc = "https://raw.githubusercontent.com/Shai2u/RI_Microsimulator_July2021/main/assets/legacy_resources/result_dec_21_955.csv"
jsonBldgs = gpd.read_file(geoJSONloc, driver='GeoJSON').to_crs("EPSG:4326")
resData = pd.read_csv(csvResultsloc)
resData = resData.iloc[:, 1:]
resData['birth_date'] = pd.to_datetime(resData['birth_date'], errors='coerce')
resData = resData[resData['birth_date'].notna()].copy()
resData.loc[:, 'move_in'] = resData.loc[:, 'move_in'].apply(pd.to_datetime)
resData.loc[:, 'move_out'] = resData.loc[:, 'move_out'].apply(pd.to_datetime)


incomeSmallCat = group_color_dict[[
    'income', 'income min', 'income max']].drop_duplicates()
incomeSmallCat.set_index('income', inplace=True)
ageSmallCat = group_color_dict[['Age Group',
                                'Age min', 'Age max']].drop_duplicates()
ageSmallCat.set_index('Age Group', inplace=True)
# ageSmallCat['Age max'] =[45,65,85,150]
ageRangeSmall = [0]+ageSmallCat['Age max'].values.tolist()
ageLbaelsSmall = ageSmallCat.index.tolist()
incomeRangeSmall = incomeSmallCat['income max'].values.tolist() + [0]
incomeLabelSmall = incomeSmallCat.index.tolist()
incomeRangeSmall.reverse()
incomeLabelSmall.reverse()
years_list = np.arange(1970, 2070, 15)
yearListSlider = dict(zip(years_list, (str(year_) for year_ in years_list)))
fontSize = 24
yearListSlider = {
    1970: {'label': '1970', 'style': {'font-size': f'{fontSize}px'}},
    1985: {'label': '1985', 'style': {'font-size': f'{fontSize}px'}},
    2000: {'label': '2000', 'style': {'font-size': f'{fontSize}px'}},
    2015: {'label': '2015', 'style': {'font-size': f'{fontSize}px'}},
    2030: {'label': '2030', 'style': {'font-size': f'{fontSize}px'}},
    2045: {'label': '2045', 'style': {'font-size': f'{fontSize}px'}},
    2060: {'label': '2060', 'style': {'font-size': f'{fontSize}px'}},
}

affordColor = color_labels[color_labels['label']
                           == 'Affordable']['colors_'].values[0]
marketColor = color_labels[color_labels['label']
                           == 'Market']['colors_'].values[0]


class bldFunctionality():
    def __init__(self):
        pass

    def filterByFullYear(self, year_):
        sy = pd.to_datetime(str(year_)+"-1-1")
        eoy = pd.to_datetime(str(year_+1)+"-1-1")  # end of year

        sample_ds = self.ds.query(
            f'move_in<="{sy}" and move_out>="{eoy}"').copy()
        return sample_ds

    def filterDisplacedByFullYearDate(self, year_):
        ds = self.ds.copy()
        sy = pd.to_datetime(str(year_)+"-1-1")
        eoy = pd.to_datetime(str(year_+1)+"-1-1")  # end of year

        step_1 = self.ds.query(f'move_in<="{sy}" and move_out>"{sy}"').copy()
        sample_ds = step_1.query(f'move_out<"{eoy}"').copy()
        return sample_ds

    def getReportDummies(self, year):
        yearDS = self.ds[['birth_date', 'Group', 'Building Name', 'annual_expenses_burden',
                          'income', 'agentID', 'tenant_cycle', 'death_age', 'entrance_age', 'ApartmentType']].copy()
        yearDS['raw age'] = year - yearDS['birth_date'].dt.year
        yearDS['get_mid'] = yearDS['raw age'].apply(ageClass.getGroupCategory)
        yearDS['age_group'] = yearDS['get_mid'].apply(
            lambda x: ageClass.mid2Group[x])
        yearDS['annual_expenses_burden'] = yearDS['annual_expenses_burden'].fillna(
            0)
        yearDS['income_group'] = yearDS['income'].apply(
            incomeClass.getIncomeCategory)
        yearDS['year'] = year
        yearDS_Dummy = pd.get_dummies(yearDS, columns=[
                                      'age_group', 'income_group', 'ApartmentType']).drop(columns=['birth_date', 'get_mid'])
        yearDS_Dummy.reset_index(inplace=True, drop=True)
        return yearDS_Dummy

    def getAffordableMarketPerYear3(self, ye_, gr=group_color_dict, gr2=censusIncomeDict):
        '''Recives Results, Year and Color File'''
        res_1 = self.filterByFullYear(ye_)
        res_2 = res_1[['Group', 'Building Name', 'ap_index',
                       'affordable_living', 'income', 'tenant_cycle', 'birth_date']].copy()
        res_2['income_group'] = res_2['income'].apply(
            incomeClass.getIncomeCategory)
        res_2['raw age'] = ye_ - res_2['birth_date'].dt.year
        res_2['get_mid'] = res_2['raw age'].apply(ageClass.getGroupCategory)
        res_2['age_group'] = res_2['get_mid'].apply(
            lambda x: ageClass.mid2Group[x])
        res_2['year'] = ye_
        res_2['ap_class'] = res_2['affordable_living'].apply(
            lambda x: 'Protected' if x == 1 else 'Market')
        res_2.reset_index(inplace=True, drop=True)
        for i in gr.index:
            row_ = gr.loc[i]
            min_age, max_age, min_income, max_income = row_['Age min'], row_[
                'Age max'], row_['income min'], row_['income max']
            age_group, income_group = row_['Age Group'], row_['income']
            gr_name = row_['group_name']
            color_ = row_['colors_']
            age_color = row_['colors_age']
            income_color = row_['colors_income']
            search_ = ((res_2['raw age'] >= min_age) & (res_2['raw age'] <= max_age)) & (
                (res_2['income'] >= min_income) & (res_2['income'] <= max_income))
            # search_ =
            res_2.loc[search_, 'group_name'] = gr_name
            res_2.loc[search_, 'group_color'] = color_
            res_2.loc[search_, 'AgeGroup4'] = age_group
            res_2.loc[search_, 'IncomeGroup4'] = income_group
            res_2.loc[search_, 'age_color'] = age_color
            res_2.loc[search_, 'income_color'] = income_color
        for item in gr2.items():
            res_2.loc[res_2['income_group'] == item[0],
                      'icnome_color_census'] = item[1]
        res_2.loc[res_2['ap_class'] == 'Protected', 'ap_class'] = 'Affordable'

        res_2.rename(columns={'IncomeGroup4': 'Income', 'AgeGroup4': 'Age Group',
                              'tenant_cycle': 'Tenant Cycle', 'ap_index': 'Door Number'}, inplace=True)
        return res_2

    @staticmethod
    def addPercentToReport1(rp):
        cols_1 = rp.columns[(rp.columns.str.contains("age_group") | rp.columns.str.contains("income_group") |
                             rp.columns.str.contains("ApartmentType_"))].tolist()
        newCols1 = [item+"_%" for item in cols_1]
        for i in range(len(cols_1)):
            sCol = cols_1[i]
            nCol = newCols1[i]
            total = rp.loc[0, 'agentID']
            colValue = rp.loc[0, sCol]
            calcRatio = colValue/total
            rp.loc[0, nCol] = round(np.mean(calcRatio), 2)
        return rp

    @staticmethod
    def dummyReportGroupByBldg(dr):
        demo_eco_cols = dr.columns[(dr.columns.str.contains("age_group") | dr.columns.str.contains("income_group") |
                                    dr.columns.str.contains("ApartmentType_"))].tolist()
        basce_cols_dict = {'Group': 'first', 'raw age': lambda x: round(np.mean(x), 1), 'income': lambda x: round(np.mean(x), 1), 'annual_expenses_burden': lambda x: round(np.mean(
            x), 1), 'agentID': 'count', 'tenant_cycle': lambda x: round(np.mean(x), 3), 'death_age': lambda x: round(np.mean(x), 1), 'entrance_age': lambda x: round(np.mean(x), 1), 'year': 'first'}
        demo_eco_cols_dict = {item: 'sum' for item in demo_eco_cols}
        base_demo_eco_dict = basce_cols_dict
        base_demo_eco_dict.update(demo_eco_cols_dict)
        # Function for each type of aggregation (No Aggregation, group or building)
        # Case og Building
        report_1 = dr.groupby('Building Name').agg(
            base_demo_eco_dict).reset_index()
        return report_1

    def rerturnDemoEcoReportForYear(self, year_, gropBy_):
        '''
        groupBy_ can be: 'Group,Building,None'
        year...
        '''
        if (gropBy_ == 'Building'):
            fullYearDS = bldDataset(self.filterByFullYear(year_).copy())
            if (len(fullYearDS.ds)) > 0:
                dummyReport = fullYearDS.getReportDummies(year_)
                report = bldFunctionality.addPercentToReport1(
                    bldFunctionality.dummyReportGroupByBldg(dummyReport)).copy()
            else:
                return None
        else:
            return None
        return report

    def yearlyReportBldgDemoEco(self):
        report_bld_years = pd.DataFrame()  # new Year
        years = np.arange(1975, 2071, 1)
        for ye_ in years:
            if np.mod(ye_, 10) == 0:
                print('year:', ye_)
            report = self.rerturnDemoEcoReportForYear(ye_, 'Building')
            if report is not None:
                if ye_ == 1975:
                    report_bld_years = report.copy()
                else:
                    report_bld_years = pd.concat(
                        [report_bld_years, report.copy()])
            else:
                continue
        return report_bld_years


class simDataset(bldFunctionality):
    def __init__(self, ds, js_bldgs, option='default'):
        #         ds['birth_date'] = pd.to_datetime(ds['birth_date'],errors='coerce')
        #         ds = ds[ds['birth_date'].notna()].copy()
        #         ds.loc[:,'move_in'] = ds.loc[:,'move_in'].apply(pd.to_datetime)
        #         ds.loc[:,'move_out'] = ds.loc[:,'move_out'].apply(pd.to_datetime)
        if option == 'WIRE':
            ds = ds[ds['Group'] == 'WIRE'].copy()
            js_bldgs = js_bldgs[js_bldgs['Group'] == 'WIRE'].copy()
        bldFunctionality.__init__(self)
        self.ds = ds

        self.json_bldgs = js_bldgs
        self.json_bldgs_slim = js_bldgs[[
            'Bldg Proje', 'bld_key', 'cnstrct_yr', 'Total Unit', 'Category', 'Group', 'geometry']]
        self.bldgs_list = js_bldgs['Bldg Proje'].unique()
        self.bldgs_ids = js_bldgs['bldgs_id'].unique()
        self.bldg_dict = self.json_bldgs[[
            'bldgs_id', 'cnstrct_yr', 'Bldg Proje', 'Category', 'Group']]
        self.bldg_bld_group = self.getBldGroup()
        bg = self.bldg_bld_group
        self.bldgsGroupDict = dict(zip(bg['Buildings'], bg['Group']))
        gb = dict(zip(bg['Buildings'], bg['Group']))  # Reverse Dict
        self.GroupbldgDict = {
            v1: [k1 for k1, v2 in gb.items() if v1 == v2] for v1 in gb.values()}

    def getBldGroup(self):
        bldg_year_ds = self.bldg_dict[['Bldg Proje', 'Group']].drop_duplicates(
        ).sort_values(by='Group').reset_index(drop=True)
        bldg_year_ds.rename(columns={'Bldg Proje': 'Buildings'}, inplace=True)
        bldg_year_ds = bldg_year_ds[bldg_year_ds['Buildings']
                                    != 'The House'].reset_index(drop=True)
        return bldg_year_ds


class bldDataset(bldFunctionality):
    # class bldDataset:
    def __init__(self, ds):
        bldFunctionality.__init__(self)
        self.ds = ds
        self.bldgs_list = ds['Building Name'].unique().tolist()


class ageClass:
    ageGroup2Mid = {'25-35': 30,
                    '36-45': 40,
                    '46-55': 50,
                    '56-60': 58,
                    '61-65': 63,
                    '66-75': 70,
                    '76-85': 80,
                    '90+': 90}
    mid2Group = {value: key for (key, value) in ageGroup2Mid.items()}

    def getGroupCategory(age_):
        """Returns the group Category for a given Age"""
        if (age_ > 85):
            return 90
        elif (age_ > 75):
            return 80
        elif (age_ > 65):
            return 70
        elif (age_ > 60):
            return 63
        elif (age_ > 55):
            return 58
        elif (age_ > 45):
            return 50
        elif(age_ > 35):
            return 40
        else:
            return 30


# In[7]:


class incomeClass:
    def getIncomeCategory(x):
        """Returns a income category for a given income"""
        # get tiltes for given income
        # need to add a sort function
        if (x >= 200000):
            return '$200K+'
        elif (x >= 150000):
            return '$150K-199K'
        elif (x >= 100000):
            return '$100K-149K'
        elif (x >= 75000):
            return '$75K-99K'
        elif (x >= 50000):
            return '$50K-74K'
        elif (x >= 35000):
            return '$35K-49K'
        elif (x >= 25000):
            return '$25K-34K'
        elif (x >= 15000):
            return '$15K-24K'
        else:
            return '<$15K'

    mid_income = {'$200K+': 225000,
                  '$150K-199K': 175000,
                  '$100K-149K': 125000,
                  '$75K-99K': 80000,
                  '$50K-74K': 60000,
                  '$35K-49K': 40000,
                  '$25K-34K': 30000,
                  '$15K-24K': 20000,
                  '<$15K': 7500}


catIncome = list(censusIncomeDict.keys())
catIncome.reverse()


class sim_plot:
    tl_width = 1250*1.5
    tl_height = 600 *1.2
    tl_height2 = 600*1.5
    cont_width = 600*1.5
    mapWidth = 600
    mapHeight1 = tl_height2*2-30
    textSize_ = 24
    @staticmethod
    def treeMapBuilding(r, titleText_):
        fig = px.treemap(r, path=['Group', 'Building Name', 'ap_class', 'Door Number'],
                         color='income', color_continuous_scale='oranges', title=titleText_)
        fig.update_layout(margin=dict(l=50, r=50, t=100, b=50),
                          width=sim_plot.cont_width, height=sim_plot.tl_height,font=dict(size=sim_plot.textSize_))

        return fig

    def treeMapIsland(r, titleText_):
        fig = px.treemap(r, path=['Group', 'Building Name', 'ap_class'],
                         color='income', title=titleText_, color_continuous_scale='oranges')
        fig.update_layout(margin=dict(l=50, r=50, t=100, b=50),
                          width=sim_plot.cont_width, height=sim_plot.tl_height,font=dict(size=sim_plot.textSize_))

        return fig

    @staticmethod
    def reasulToLeaveByTime(r, titleText_):
        r = r[['Building Name', 'Group', 'cause',
               'stay_go', 'agentID', 'year']].copy()
        r['status'] = r['stay_go']
        r.loc[r['stay_go'] == 'out',
              'status'] = r.loc[r['stay_go'] == 'out', 'cause']
        r2 = r.groupby(['year', 'status']).agg(
            {'agentID': 'count'}).reset_index()
        r2.rename(columns={'agentID': 'Household Agents'}, inplace=True)
        r2 = r2[r2['status'].isin(
            ['Rent Burden', 'death', 'Mortgage Burden', 'Total Burden'])]

        r2.loc[r2['status'] == 'death', 'status'] = 'Death'

        leaveColorDict = {'Rent Burden': 'blue', 'Death': 'red',
                          'Mortgage Burden': 'purple', 'Total Burden': 'green'}

        fig = px.bar(r2, x="year", y="Household Agents",
                     color="status", color_discrete_map=leaveColorDict, title=titleText_, template='plotly_white')
        fig.update_layout(margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.tl_width, height=sim_plot.tl_height2, legend=dict(
            yanchor="top", y=0.9, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",font=dict(size=sim_plot.textSize_))

        return fig

    @staticmethod
    def averageAgeByTime(r, titleText_):
        r = r[['Building Name', 'Group', 'death_age', 'raw age',
               'income', 'annual_expenses_burden', 'year']].copy()
        r2 = r.groupby(['year']).agg({'raw age': lambda x: round(
            x.mean(), 0), 'death_age': lambda x: round(x.mean(), 0)}).reset_index()
        r2.rename(columns={'agentID': 'Household Agents',
                           'raw age': 'Mean Age', 'death_age': 'Death Age'}, inplace=True)

        fig = px.line(r2, x="year", y=["Mean Age", "Death Age"], title=titleText_,
                      template='plotly_white', labels=dict(value="Age", variable="Legend"))
        fig.update_layout(margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.tl_width, height=sim_plot.tl_height2, legend=dict(
            yanchor="top", y=0.9, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",font=dict(size=sim_plot.textSize_))

        return fig

    @staticmethod
    def ageGroupTimeGraph(r, titleText_):
        r = r[['Building Name', 'Group', 'age_group_2', 'agentID', 'year']].copy()
        r2 = r.groupby(['year', 'age_group_2']).agg(
            {'agentID': 'count'}).reset_index()
        r2.rename(columns={'agentID': 'Household Agents',
                           'age_group_2': 'Age Group'}, inplace=True)

        fig = px.line(r2, x="year", y="Household Agents",
                      color="Age Group", color_discrete_map=colorDict, title=titleText_, template='plotly_white')
        fig.update_layout(margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.tl_width, height=sim_plot.tl_height2, legend=dict(
            yanchor="top", y=1.05, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",font=dict(size=sim_plot.textSize_))

        return fig

    @staticmethod
    def incomeGroupTimeGraph(r, titleText_):
        r = r[['Building Name', 'Group', 'income_group_2', 'agentID', 'year']].copy()
        r2 = r.groupby(['year', 'income_group_2']).agg(
            {'agentID': 'count'}).reset_index()
        r2.rename(columns={'agentID': 'Household Agents',
                           'income_group_2': 'Income Group'}, inplace=True)

        fig = px.line(r2, x="year", y="Household Agents", color="Income Group",
                      color_discrete_map=colorDict, title=titleText_, template='plotly_white')

        fig.update_layout(margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.tl_width, height=sim_plot.tl_height2, legend=dict(
            yanchor="top", y=0.9, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",font=dict(size=sim_plot.textSize_))

        return fig

    @staticmethod
    def incomeBurdenTime(r, titleText_):
        r = r[['Building Name', 'Group', 'income',
               'annual_expenses', 'year']].copy()
        r2 = r.groupby(['year']).agg({'income': lambda x: round(
            x.mean(), 0), 'annual_expenses': lambda x: round(x.mean(), 1)}).reset_index()
        r2.rename(columns={'income': 'Mean Income',
                           'annual_expenses': 'Man Annual Exprense'}, inplace=True)

        fig = px.line(r2, x="year", y=["Mean Income", "Man Annual Exprense"], title=titleText_,
                      template='ggplot2', labels=dict(value="US Dollars", variable="Expenses"))
        fig.update_layout(margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.tl_width, height=sim_plot.tl_height2, legend=dict(
            yanchor="top", y=0.9, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",font=dict(size=sim_plot.textSize_))

        return fig

    @staticmethod
    def ageByGroupFigure(r, year_, titleText_):
        r2 = r.groupby('Age Group').agg({'get_mid': 'count'}).reset_index().rename(
            columns={'get_mid': 'Households'})
        title_ = str(year_)+' '+titleText_ + ' Age Groups'
        fig = px.bar(r2, x='Age Group', y='Households', template='plotly_white', title=title_, color='Age Group',
                     color_discrete_map=colorDict, category_orders={'Age Group': ['18-44', '45-64', '65-84', '85+']})
        fig.update_layout(showlegend=False, margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.cont_width, height=sim_plot.tl_height,font=dict(size=sim_plot.textSize_))
        return fig

    @staticmethod
    def incomeByGroupFigure(r, year_, titleText_):
        r2 = r.groupby('Income').agg({'get_mid': 'count'}).reset_index().rename(
            columns={'Income': 'Income Group', 'get_mid': 'Households'})
        title_ = str(year_)+' '+titleText_ + ' Income Groups'
        fig = px.bar(r2, x='Income Group', y='Households', template='plotly_white', title=title_, color='Income Group',
                     color_discrete_map=colorDict, category_orders={'Income Group': ['Low', 'Moderate', 'Middle', 'Upper']})
        fig.update_layout(showlegend=False, margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.cont_width, height=sim_plot.tl_height,font=dict(size=sim_plot.textSize_))
        return fig

    @staticmethod
    def incomeByGroupFigureCensus(r, year_, titleText_):
        catIncome = list(censusIncomeDict.keys())
        catIncome.reverse()
        r2 = r.groupby('income_group').agg({'get_mid': 'count'}).reset_index().rename(
            columns={'income_group': 'Income Group Census Categories', 'get_mid': 'Households'})
        title_ = str(year_)+' '+titleText_ + ' Income Group Census Categories'
        fig = px.bar(r2, x='Income Group Census Categories', y='Households', template='plotly_white', title=title_,
                     color='Income Group Census Categories', color_discrete_map=censusIncomeDict, category_orders={'Income Group Census Categories': catIncome})
        fig.update_layout(showlegend=False, margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.cont_width, height=sim_plot.tl_height,font=dict(size=sim_plot.textSize_))
        return fig

    @staticmethod
    def bubbleAgeIncomeClass(r, year_, titleText_):
        r2 = r.groupby(['ap_class', 'group_name', 'Age Group', 'Income']).agg(
            {'raw age': 'count'}).reset_index().rename(columns={'raw age': 'count', 'ap_class': 'Ap Type'})

        title_ = str(year_)+' '+titleText_ + ' Age/Income'

        fig = px.scatter(r2, x="Age Group", y="Income",
                         size="count", color="group_name", color_discrete_map=colorDictMerge, facet_col='Ap Type', title=title_, size_max=30,
                         category_orders={"Age Group": ["18-44", "45-64", "65-84", "85+"],
                                          "Income": ['Upper', 'Middle', 'Moderate', 'Low']}, template='ggplot2')
        fig.update_layout(showlegend=False, margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.cont_width, height=sim_plot.tl_height,font=dict(size=sim_plot.textSize_))

        return fig

    @staticmethod
    def sunburstGroupsAffordMarketYearColor3(r, year_, color_field='raw age', colorDict_=colorDict, group_color_dict=group_color_dict):
        title_ = f'{year_} : Market Vs Affordable Units Age/Income in WIRE'
        # color_discrete_map = color_group_map_
        colors_ = group_color_dict['colors_age'].unique().tolist()
        fig = px.sunburst(r, path=['ap_class', 'Age Group', 'Income'],
                          color=color_field, color_continuous_scale=colors_, title=title_,)

        labels_text = fig.data[0].labels.tolist()
        colorLabels = tuple(colorDict_[item] for item in labels_text)
        fig.data[0].marker.colors = colorLabels
        fig.update_traces(textinfo="label+percent entry")
        fig.update_layout(showlegend=False, margin=dict(l=50, r=50, t=100, b=50), legend=dict(
            yanchor="top", y=1, xanchor="left", x=1, orientation="h"), width=sim_plot.cont_width, height=sim_plot.tl_height+50,font=dict(size=sim_plot.textSize_))
        return fig

    @staticmethod
    def getAgentsByRangeAllGroupInOut2(ds, years_range):
        for ye_ in years_range:
            if np.mod(ye_, 10) == 0:
                print('year:', ye_)
            if ye_ == years_range[0]:
                all_years = sim_plot.getAgentYearGroupInOut(ds, ye_).copy()
            else:
                toConcat = sim_plot.getAgentYearGroupInOut(ds, ye_).copy()
                all_years = pd.concat([all_years, toConcat])
        return all_years

    def getAgentYearGroupInOut(ds, ye_):
        ds_fyear = ds.filterByFullYear(ye_)
        ds_moveout = ds.filterDisplacedByFullYearDate(ye_)
        ds_fyear = ds_fyear[['Building Name', 'Group', 'tenant_cycle', 'ApartmentType', 'comment 1', 'affordable_living', 'cause', 'move_in', 'move_out',
                             'birth_date', 'death_age', 'death_date', 'income', 'annual_expenses', 'annual_expenses_burden', 'agentID']].copy()
        ds_moveout = ds_moveout[['Building Name', 'Group', 'tenant_cycle', 'ApartmentType', 'comment 1', 'affordable_living', 'cause', 'move_in', 'move_out',
                                 'birth_date', 'death_age', 'death_date', 'income', 'annual_expenses', 'annual_expenses_burden', 'agentID']].copy()
        ds_fyear['annual_expenses_burden'] = ds_fyear['annual_expenses'] / \
            ds_fyear['income']
        ds_moveout['annual_expenses_burden'] = ds_moveout['annual_expenses'] / \
            ds_moveout['income']
        ds_moveout['stay_go'] = 'out'
        ds_fyear['stay_go'] = 'stay'
        ds_fyear.loc[ds_fyear['move_in'].dt.year == ye_, 'stay_go'] = 'new'
        all_y = pd.concat([ds_fyear, ds_moveout])
        all_y['income_group'] = all_y['income'].apply(
            incomeClass.getIncomeCategory)
        all_y['raw age'] = ye_ - all_y['birth_date'].dt.year
        all_y['get_mid'] = all_y['raw age'].apply(
            ageClass.getGroupCategory)
        all_y['age_group'] = all_y['get_mid'].apply(lambda x:
                                                    ageClass.mid2Group[x])
        all_y['year'] = ye_
        return all_y

    def wireSpecificBuildingGraph(data, rental_list, coop_list, marketColor, affordColor, title_, year_):
        data = data.query(f'year<={year_}').copy()
        # buildingAggregation
        g1 = data.groupby(['year', 'Building Name', 'ApartmentType'])
        g2 = g1.agg({'agentID': 'count'}).reset_index()
        # building unique List
        bld_list = data['Building Name'].unique()
        ds_list = [g2[g2['Building Name'] == bld] for bld in bld_list]

        for i in range(len(rental_list)):
            ds_list[i].reset_index(inplace=True, drop=True)
            rental_list[i] = rental_list[i].query(f'year<={year_}').copy()
            rental_list[i].reset_index(inplace=True, drop=True)
            coop_list[i] = coop_list[i].query(f'year<={year_}').copy()
            coop_list[i].reset_index(inplace=True, drop=True)

        bld_list[2] = 'Island House'

        fig = go.Figure()
        for i in range(4):
            if (i == 0):
                line_coop = dict(color=marketColor, width=2)
                line_rent = dict(color=affordColor, width=2)
            elif (i == 1):
                line_coop = dict(color=marketColor, width=4, dash='dash')
                line_rent = dict(color=affordColor, width=4, dash='dash')
            elif (i == 2):
                line_coop = dict(color=marketColor, width=2, dash='dot')
                line_rent = dict(color=affordColor, width=2, dash='dot')
            else:
                line_coop = dict(color=marketColor, width=3, dash='dashdot')
                line_rent = dict(color=affordColor, width=3, dash='dashdot')
            rentType = ' - affordable units'
            # Percent Option
            max_c = coop_list[i]['agentID'].max()
            max_r = rental_list[i]['agentID'].max()
            percent_ = coop_list[i]['percent_']
            fig.add_trace(go.Scatter(x=coop_list[i]['year'], y=(coop_list[i]['agentID']),
                                     mode='lines',
                                     name=bld_list[i] + ' - market units',
                                     line=line_coop,
                                     legendgroup=str(i),
                                     hovertemplate='<br><b>Market Units</b>:%{y:}<br>' +
                                     '<b>Percent:</b> %{text} %',
                                     text=['{:.1f}'.format(p*100, 1)
                                           for p in percent_]
                                     ))
            percent_ = rental_list[i]['percent_']
            fig.add_trace(go.Scatter(x=rental_list[i]['year'], y=(rental_list[i]['agentID']),
                                     mode='lines',
                                     name=bld_list[i] + rentType,
                                     line=line_rent,
                                     legendgroup=str(i),
                                     hovertemplate='<br><b>Affordable Units</b> : %{y:}<br>' +
                                     '<b>Percent : </b>%{text} %',
                                     text=['{:.1f}'.format(p*100, 1)
                                           for p in percent_],
                                     ))

        fig.update_layout(title=title_)
        fig.update_xaxes(range=[1976, 2076], showline=True,
                         linecolor='rgb(150,150,150)')
        fig.update_yaxes(range=[0, 1100], showline=True,
                         linecolor='rgb(150,150,150)')
        fig.update_layout(width=sim_plot.tl_width, height=sim_plot.tl_height2, plot_bgcolor='rgba(255,255,255,0)', legend=dict(yanchor="top", y=0.8, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",
                          margin=dict(l=50, r=50, t=100, b=50), showlegend=False,font=dict(size=sim_plot.textSize_))
        return fig

    @staticmethod
    def wireGeneralGraph(aw, mC, aC):
        fig = go.Figure()
        line_coop = dict(color=mC, width=3)
        line_rent = dict(color=aC, width=3)
        percent_c = aw['Market Percent']
        percent_r = aw['Affordable Percent']
        fig.add_trace(go.Scatter(x=aw['year'], y=(aw['Market Units']),
                                 mode='lines',
                                 name='Market units',
                                 line=line_coop,
                                 hovertemplate='<br><b>Market Units</b>:%{y:}<br>' +
                                 '<b>Percent:</b> %{text} %',
                                 text=['{:.1f}'.format(p*100, 1)
                                       for p in percent_c]
                                 ))

        fig.add_trace(go.Scatter(x=aw['year'], y=(aw['Affordable Units']),
                                 mode='lines',
                                 name='Affordable units',
                                 line=line_rent,
                                 hovertemplate='<br><b>Affordable Units</b> : %{y:}<br>' +
                                 '<b>Percent : </b>%{text} %',
                                 text=['{:.1f}'.format(p*100, 1)
                                       for p in percent_r],
                                 ))

        fig.update_xaxes(range=[1976, 2076], showline=True,
                         linecolor='rgb(150,150,150)')
        fig.update_yaxes(range=[0, 2250], showline=True,
                         linecolor='rgb(150,150,150)')
        fig.update_layout(width=sim_plot.tl_width, height=sim_plot.tl_height2, plot_bgcolor='rgba(255,255,255,0)', legend=dict(yanchor="top", y=0.4, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",
                          margin=dict(l=50, r=50, t=100, b=50), title='Privatization process conversion of affordable units to market rate',font=dict(size=sim_plot.textSize_))
        return fig

    @staticmethod
    def allGeneralGraph(isam, mC, aC):
        fig = go.Figure()
        line_coop = dict(color=mC, width=3)
        line_rent = dict(color=aC, width=3)
        percent_c = isam['Market Percent']
        percent_r = isam['Affordable Percent']
        fig.add_trace(go.Scatter(x=isam['year'], y=(isam['Market Units']),
                                 mode='lines',
                                 name='Market units',
                                 line=line_coop,
                                 hovertemplate='<br><b>Market Units</b>:%{y:.0f}<br>' +
                                 '<b>Percent:</b> %{text} %',
                                 text=['{:.1f}'.format(p*100, 1)
                                       for p in percent_c]
                                 ))

        fig.add_trace(go.Scatter(x=isam['year'], y=(isam['Affordable Units']),
                                 mode='lines',
                                 name='Affordable units',
                                 line=line_rent,
                                 hovertemplate='<br><b>Affordable Units</b> : %{y:.0f}<br>' +
                                 '<b>Percent : </b>%{text} %',
                                 text=['{:.1f}'.format(p*100, 1)
                                       for p in percent_r],
                                 ))

        fig.update_layout(
            title='Privatization process conversion of affordable units to market rate')

        fig.update_layout(width=sim_plot.tl_width, height=sim_plot.tl_height2, plot_bgcolor='rgba(255,255,255,0)', hoverlabel_align="auto", hovermode="x unified", legend=dict(yanchor="top", y=0.8, xanchor="left", x=0.01),
                          margin=dict(l=50, r=50, t=100, b=50),font=dict(size=sim_plot.textSize_))
        fig.update_xaxes(range=[1976, 2076], showline=True,
                         linecolor='rgb(150,150,150)')
        #                 showgrid=True, gridcolor='rgb(150,150,150)')
        fig.update_yaxes(range=[0, 5400], showline=True,
                         linecolor='rgb(150,150,150)')

        return fig

    def AgentCycle(r, year_, title_):
        affordCycles = r.query('ap_class=="Affordable"')['Tenant Cycle']
        marketCycles = r.query('ap_class=="Market"')['Tenant Cycle']
        histogramFig = go.Figure()
        histogramFig.add_trace(go.Histogram(
            x=affordCycles, name='Affordable Units'))
        histogramFig.add_trace(go.Histogram(
            x=marketCycles, name='Market Units'))

        # Overlay both histograms
        histogramFig.update_layout(barmode='overlay', title=f'Tenant Cycles {year_} {title_}', template='plotly_white', legend=dict(
            yanchor="top", y=0.85, xanchor="left", x=0.01, orientation="h"), margin=dict(l=50, r=50, t=100, b=50), width=sim_plot.cont_width, height=sim_plot.tl_height)
        # Reduce opacity to see both histograms
        histogramFig.update_traces(opacity=0.75)
        return histogramFig


resData.loc[resData['Building Name'] ==
            'island house', 'Building Name'] = 'Island House'
jsonBldgs.loc[jsonBldgs['Bldg Proje'] ==
              'island house', 'Bldg Proje'] = 'Island House'
jsonBldgs = jsonBldgs.loc[jsonBldgs['Bldg Proje'] != 'The House']
jsonBldgs['Buildings'] = jsonBldgs['Bldg Proje']
rib = jsonBldgs.copy()
resultsAll1 = simDataset(resData, jsonBldgs)


allByYear = sim_plot.getAgentsByRangeAllGroupInOut2(
    resultsAll1, range(1976, 2080))


# In[12]:


allByYear['age_group_2'] = pd.cut(
    allByYear['raw age'].values, ageRangeSmall, labels=ageLbaelsSmall)
allByYear['income_group_2'] = pd.cut(
    allByYear['income'].values, incomeRangeSmall, labels=incomeLabelSmall)
# allByYear.groupby('year')
allByYearStay = allByYear[allByYear['stay_go'] != 'out'].copy().reset_index()
# Get a WIRE subset
allByYearWire = allByYearStay.loc[allByYearStay['Group'] == 'WIRE'].copy()
allByYearWire.reset_index(inplace=True, drop=True)

allByYearNotWire = allByYearStay.loc[allByYearStay['Group'] != 'WIRE'].copy()
allByYearNotWire.reset_index(inplace=True, drop=True)


def addAffordableStatistics(r):
    r_dummy = pd.get_dummies(r[['year', 'affordable_living']], columns=[
                             'affordable_living'])
    r_dummy_agg = r_dummy.groupby('year').agg('sum').reset_index()
    r_dummy_agg = r_dummy_agg.rename(columns={
                                     'affordable_living_0': 'Market units', 'affordable_living_1': 'Affordable units'})
    r_dummy_agg['total'] = r_dummy_agg['Market units'] + \
        r_dummy_agg['Affordable units']
    r_dummy_agg['Market Percent'] = r_dummy_agg['Market units'] / \
        r_dummy_agg['total']
    r_dummy_agg['Affordable Percent'] = r_dummy_agg['Affordable units'] / \
        r_dummy_agg['total']
    return r_dummy_agg


def addAffordableStatisticsByBuildings(r):
    r_dummy = pd.get_dummies(
        r[['year', 'Group', 'Building Name', 'affordable_living']], columns=['affordable_living'])
    r_dummy_agg = r_dummy.groupby(
        ['Group', 'Building Name', 'year']).agg('sum').reset_index()
    r_dummy_agg = r_dummy_agg.rename(columns={
                                     'affordable_living_0': 'Market units', 'affordable_living_1': 'Affordable units'})
    r_dummy_agg['total'] = r_dummy_agg['Market units'] + \
        r_dummy_agg['Affordable units']
    r_dummy_agg['Market Percent'] = r_dummy_agg['Market units'] / \
        r_dummy_agg['total']
    r_dummy_agg['Affordable Percent'] = r_dummy_agg['Affordable units'] / \
        r_dummy_agg['total']
    return r_dummy_agg


allAffordableByBldg = addAffordableStatisticsByBuildings(allByYear)
NotWireAffordalbe = addAffordableStatistics(allByYearNotWire)
AllAffordable = addAffordableStatistics(allByYear)
WireAffordable = addAffordableStatistics(allByYearWire)


def affordabilityTimeSeriesAgregattedGraph(aw, mC, aC, title_):
    bldgsGroupTitle = title_
    fig = go.Figure()
    line_coop = dict(color=mC, width=3)
    line_rent = dict(color=aC, width=3)
    percent_c = aw['Market Percent']
    percent_r = aw['Affordable Percent']
    fig.add_trace(go.Scatter(x=aw['year'], y=(aw['Market units']),
                             mode='lines',
                             name='Market units',
                             line=line_coop,
                             hovertemplate='<br><b>Market Units</b>:%{y:}<br>' +
                             '<b>Percent:</b> %{text} %',
                             text=['{:.1f}'.format(p*100, 1)
                                   for p in percent_c]
                             ))

    fig.add_trace(go.Scatter(x=aw['year'], y=(aw['Affordable units']),
                             mode='lines',
                             name='Affordable units',
                             line=line_rent,
                             hovertemplate='<br><b>Affordable Units</b> : %{y:}<br>' +
                             '<b>Percent : </b>%{text} %',
                             text=['{:.1f}'.format(p*100, 1)
                                   for p in percent_r],
                             ))

    fig.add_trace(go.Scatter(x=aw['year'], y=(aw['total']),
                             mode='lines',
                             name='Total Units',
                             line=dict(color='hsl(30, 96%, 74%)',
                                       width=2, dash='dash'),
                             hovertemplate='<br><b>Total Units</b> : %{y:}<br>'
                             ))
    maxY = aw['total'].max()
    maxY *= 1.25

    fig.update_xaxes(range=[1976, 2076], showline=True,
                     linecolor='rgb(150,150,150)', title='Year')
    fig.update_yaxes(range=[0, maxY], showline=True,
                     linecolor='rgb(150,150,150)', title='Households')
    fig.update_layout(width=sim_plot.tl_width, height=sim_plot.tl_height2, plot_bgcolor='rgba(255,255,255,0)', legend=dict(yanchor="top", y=0.97, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",
                      margin=dict(l=50, r=50, t=100, b=50), title=bldgsGroupTitle + " Market and Affordable Units Time Series",font=dict(size=sim_plot.textSize_))
    return fig


def baseFigForAffordableTS(title_, ymax, yloc):
    fig = go.Figure()
    fig.update_xaxes(range=[1976, 2076], showline=True,
                     linecolor='rgb(150,150,150)', title='Year')
    fig.update_yaxes(range=[0, ymax], showline=True,
                     linecolor='rgb(150,150,150)', title='Households')
    fig.update_layout(width=sim_plot.tl_width, height=sim_plot.tl_height2, plot_bgcolor='rgba(255,255,255,0)', legend=dict(yanchor="top", y=yloc, xanchor="left", x=0.01, orientation="h"), hoverlabel_align="auto", hovermode="x unified",
                     margin=dict(l=50, r=50, t=100, b=50), title=title_ + " Market and Affordable Units Time Series",font=dict(size=sim_plot.textSize_))
    return fig


def baseFigForAffordableTSAddTrace(fig, r, colx, coly, line_style, hovertext_, name_):
    fig.add_trace(go.Scatter(x=r[colx], y=(r[coly]),
                             mode='lines',
                             name=name_,
                             line=line_style,
                             hovertemplate=hovertext_,
                             legendgroup=name_))
    return fig


def createAffordableInduvidualBldgs(title_, maxY, yloc, query_):
    fig1 = baseFigForAffordableTS(title_, maxY, yloc)
    r1 = allAffordableByBldg.query(query_)
    BldgsList = r1.query(query_)['Building Name'].unique().tolist()
    lineDashl = ['Line', 'dash', 'dot', 'dashdot']
    for i in range(len(BldgsList)):
        bldg = BldgsList[i]
        if i == 0:
            line_style_ = dict(color=marketColor, width=i+1)
        elif i in [1, 4, 8, 12]:
            line_style_ = dict(color=marketColor, width=i+1, dash=lineDashl[1])
        elif i in [2, 5, 9, 13]:
            line_style_ = dict(color=marketColor, width=i+1, dash=lineDashl[2])
        else:
            line_style_ = dict(color=marketColor, width=i+1, dash=lineDashl[3])

        r = r1[r1['Building Name'] == bldg].copy()
        r.reset_index(inplace=True, drop=True)

        hovertext_ = [
            f'<br><b>Market Units</b>:{r.loc[i,"Market units"]}<br><b>Percent:</b> {"{:.1%}".format(r.loc[i,"Market Percent"])}' for i in range(len(r))]
        # ,text = ['{:.1f}'.format(p*100,1) for p in r['Market Percent']]
        fig1 = baseFigForAffordableTSAddTrace(
            fig1, r, colx='year', coly='Market units', line_style=line_style_, hovertext_=hovertext_, name_=bldg)
    r = r1.copy()
    for i in range(len(BldgsList)):
        bldg = BldgsList[i]
        if i == 0:
            line_style_ = dict(color=affordColor, width=i+1)
        elif i in [1, 4, 8, 12]:
            line_style_ = dict(color=affordColor, width=i+1, dash=lineDashl[1])
        elif i in [2, 5, 9, 13]:
            line_style_ = dict(color=affordColor, width=i+1, dash=lineDashl[2])
        else:
            line_style_ = dict(color=affordColor, width=i+1, dash=lineDashl[3])
        r = r1[r1['Building Name'] == bldg].copy()
        r.reset_index(inplace=True, drop=True)
        # line_style_ = dict(color=affordColor, width=i+1)
#         r = allAffordableByBldg[allAffordableByBldg['Building Name']==bldg].copy()
#         r.reset_index(inplace=True,drop=True)

        hovertext_ = [
            f'<br><b>Affordable Units</b>:{r.loc[i,"Affordable units"]}<br><b>Percent:</b> {"{:.1%}".format(r.loc[i,"Affordable Percent"])}' for i in range(len(r))]
        # ,text = ['{:.1f}'.format(p*100,1) for p in r['Market Percent']]
        fig1 = baseFigForAffordableTSAddTrace(
            fig1, r, colx='year', coly='Affordable units', line_style=line_style_, hovertext_=hovertext_, name_=bldg)
    return fig1


def getCurrentScope(r):
    uniqBldgs = r['Building Name'].nunique()
    agentsC = len(r)
    tenantCycle = round(r['Tenant Cycle'].mean(), 1)
    meanAge = round(r['raw age'].mean(), 1)
    meanIncome = round(r['income'].mean(), 0)
    rtText = f"In Scope - Buildings: {uniqBldgs}, HH: {agentsC}, Tenant Cycles: {tenantCycle} Mean Age:{meanAge}, Mean Income:{meanIncome}$"
    return rtText


def getIframeURLfor3D(zoomto='All of The Island'):
    buildingsIframeDict = {
        'base': 'https://technion-gis.maps.arcgis.com/apps/instant/3dviewer/index.html?appid=70a5849b08a643e188c1e082cfb579c4',
        'Roosevelt Landings': '&viewpoint=cam:-8232294.14564431,4976971.55542625,188.707,102100;53.45,62.097',
        'Manhattan park': '&viewpoint=cam:-8231926.10204526,4978087.15207409,101.822,102100;149.074,73.52',
        'The Octagon': '&viewpoint=cam:-8231799.25126456,4978433.16302919,116.932,102100;110.548,74.849',
        'Island House': '&viewpoint=cam:-8232262.63739977,4977617.48364878,131.456,102100;153.421,69.801',
        'Rivercross': '&viewpoint=cam:-8232476.22640797,4977191.75742142,116.144,102100;88.42,71.129',
        'Riverwalk Landing': '&viewpoint=cam:-8232744.92622803,4976927.06005498,135.343,102100;127.737,70.598',
        'Riverwalk Point': '&viewpoint=cam:-8232254.75160611,4976616.19499,129.249,102100;5.637,65.816',
        'Westview': '&viewpoint=cam:-8232152.701749,4977766.89078352,122.765,102100;153.421,69.801',
        'Riverwalk place': '&viewpoint=cam:-8232599.54300243,4976981.47710137,178.258,102100;97.706,58.909',
        'Riverwalk Court': '&viewpoint=cam:-8232748.95180919,4976914.78684295,129.03,102100;111.238,71.718',
        'Riverwalk Crossing': '&viewpoint=cam:-8232771.27998685,4976891.42089825,135.343,102100;127.737,70.598',
        '2-4 River Road': '&viewpoint=cam:-8232014.67558545,4977890.22421553,101.822,102100;149.074,73.52',
        'Wire': '&viewpoint=cam:-8232455.27430039,4976611.10725427,942.614,102100;34.366,38.656',
        'All of The Island': '&viewpoint=cam:-8233187.87589454,4975509.19937045,682.802,102100;32.257,68.071',
        'Northtown & Southtown': '&viewpoint=cam:-8233187.87589454,4975509.19937045,682.802,102100;32.257,68.071'}
    return f"{buildingsIframeDict['base']}{buildingsIframeDict[zoomto]}"


def updateMapYear1(value_, rMap, r, cat='aib', zoomto='All of The Island'):

    if zoomto == 'Wire':
        focus = rib[rib['Group'].isin(['WIRE'])].copy()
        lon_ = focus.geometry.boundary.centroid.x.values[0]
        lat_ = focus.geometry.boundary.centroid.y.values[0]
        zoom_ = 16
    elif zoomto in ['All of The Island', 'Northtown & Southtown']:
        zoom_ = 14.5
        lat_ = 40.7624
        lon_ = -73.949
    else:
        focus = rib[rib['Buildings'].isin([zoomto])].copy()
        lon_ = focus.geometry.boundary.centroid.x.values[0]
        lat_ = focus.geometry.boundary.centroid.y.values[0]
        zoom_ = 17
    rib_filter = rMap[rMap['cnstrct_yr'] < value_].copy()
    # r2 = r.copy()
    mapAggData = r.groupby('Building Name').agg({'Age Group': lambda x: x.value_counts().index[0], 'Income': lambda x: x.value_counts().index[0], 'group_name': lambda x: x.value_counts(
    ).index[0], 'Tenant Cycle': lambda x: np.round(x.mean(), 1), 'income': lambda x: np.round(x.mean(), 1), 'raw age': lambda x: np.round(x.mean(), 1)}).reset_index()
    mapAggData.rename(columns={'group_name': 'Age Income',
                               'income': 'Mean Income', 'raw age': 'Mean Age'}, inplace=True)
    if ('Market' in r['ap_class'].unique().tolist()):
        affordableMarket = pd.get_dummies(r, columns=['ap_class']).groupby('Building Name').agg(
            {'ap_class_Affordable': 'sum', 'ap_class_Market': 'sum', 'Door Number': 'count'}).reset_index()
        affordableMarket['Affordable Ratio'] = affordableMarket['ap_class_Affordable'] / \
            affordableMarket['Door Number']
    else:
        affordableMarket = pd.get_dummies(r, columns=['ap_class']).groupby('Building Name').agg(
            {'ap_class_Affordable': 'sum', 'Door Number': 'count'}).reset_index()
        affordableMarket['Affordable Ratio'] = 1

    affordableMarket = affordableMarket[['Building Name', 'Affordable Ratio']]
    mapAggData = pd.merge(mapAggData, affordableMarket,
                          on='Building Name', how='left')  # Affordable Ratio
    mapAggData['Affordable Ratio'].fillna(0, inplace=True)
    # colorDictMerge
    mapModifed = pd.merge(rib_filter, mapAggData, how='right',
                          left_on='Buildings', right_on='Building Name')

    if cat == 'aib':
        colorCat = 'Age Income'
        discreteMap = colorDictMerge
    elif cat == 'income':
        colorCat = 'Income'
        discreteMap = colorDict
    elif cat == 'meanIncome':
        colorCat = 'Mean Income'
    elif cat == 'meanAge':
        colorCat = 'Mean Age'
    elif cat == 'age':
        colorCat = 'Age Group'
        discreteMap = colorDict
    elif cat == 'apNum':
        colorCat = 'Total Unit'
    elif cat == 'afPercent':
        colorCat = 'Affordable Ratio'
    else:
        colorCat = 'Tenant Cycle'

    if cat in ['cycle', 'apNum', 'afPercent', 'meanIncome', 'meanAge']:
        localCoConColorSclae = {'meanIncome': 'YlOrRd', 'meanAge': ['#e6ad00', '#fefdfc'],
                                'cycle': 'YlOrBr', 'apNum': 'YlOrBr', 'afPercent': 'YlOrBr'}
        rangeColorDict = {'meanIncome': (10000, 350000), 'meanAge': (40, 65),
                          'cycle': (0, 7), 'apNum': (50, 900), 'afPercent': (0, 1)}

        chosenScale = localCoConColorSclae[cat]
        rangeColor_ = rangeColorDict[cat]
        fig_map = px.choropleth_mapbox(mapModifed, geojson=mapModifed.geometry, locations=mapModifed.index, color=colorCat, color_continuous_scale=chosenScale,
                                       mapbox_style="carto-positron",
                                       hover_name='Bldg Proje',
                                       custom_data=['Buildings', 'Group'],
                                       range_color=rangeColor_
                                       )
        fig_map.update_traces(colorbar=dict(
            thickness=5, len=0.25, ticks='inside', showticklabels=False))  # ) #marker_showscale=False
    else:
        fig_map = px.choropleth_mapbox(mapModifed, geojson=mapModifed.geometry, locations=mapModifed.index, color=colorCat, color_discrete_map=discreteMap,
                                       mapbox_style="carto-positron",
                                       hover_name='Bldg Proje',
                                       custom_data=['Buildings', 'Group']
                                       ).update_traces(showlegend=True)
    mapbox_ = dict(bearing=33, pitch=0, zoom=zoom_,
                   center=dict(lat=lat_, lon=lon_))
    fig_map.update_layout(autosize=True, height=sim_plot.mapHeight1, width=sim_plot.mapWidth, mapbox=mapbox_, legend=dict(
        yanchor="top", y=0.1, xanchor="left", x=0.01, orientation="h"),margin=dict(l=0, r=0, t=0, b=0))

    return fig_map


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


bgcolor = "#f3f3f1"  # mapbox light map land color

bldListOrder = ['Roosevelt Landings', 'Roosevelt Landings', 'Rivercross',
                'Rivercross', 'Island House', 'Island House', 'Westview', 'Westview']

valYear = 2000
generalCopy = AllAffordable.copy()
generalCopy = generalCopy.query(f"year<={valYear}")
figAll = affordabilityTimeSeriesAgregattedGraph(
    aw=generalCopy, mC=marketColor, aC=affordColor, title_='All of the Island')

cellTimeFigure = dcc.Graph(id='time-graph', figure=figAll)
cellTimeFigureProjDash = dcc.Graph(id='time-graphProjDash', figure=figAll)
currentData = resultsAll1.getAffordableMarketPerYear3(valYear)
figSunBurst = sim_plot.sunburstGroupsAffordMarketYearColor3(
    currentData, valYear)

bubbleFig_ = sim_plot.bubbleAgeIncomeClass(currentData, valYear, 'All')
fig_map = updateMapYear1(valYear, rib.copy(), currentData,
                         'age', zoomto='All of The Island')
mapColorDropDownMenu = dcc.Dropdown(id='mapcolor-menu',
                                    options=[
                                        {'label': 'Age/Income', 'value': 'aib'},
                                        {'label': 'Income Groups',
                                            'value': 'income'},
                                        {'label': 'Mean Income ($)',
                                         'value': 'meanIncome'},
                                        {'label': 'Age Groups', 'value': 'age'},
                                        {'label': 'Mean Age', 'value': 'meanAge'},
                                        {'label': 'Apartment Cycles',
                                            'value': 'cycle'},
                                        {'label': 'Apartment Numbers',
                                            'value': 'apNum'},
                                        {'label': 'Affordable Percent',
                                            'value': 'afPercent'}
                                    ],
                                    value='aib'
                                    )
timeDropDownMenu = dcc.Dropdown(id='time-menu',
                                options=[
                                    {'label': 'Affordable Market', 'value': 'am'},
                                    {'label': 'Leaving', 'value': 'leave'},
                                    {'label': 'Life Expectancy', 'value': 'life'},
                                    {'label': 'Income Expenses', 'value': 'ie'},
                                    {'label': 'Age Groups', 'value': 'ageg'},
                                    {'label': 'Income Groups', 'value': 'ig'},
                                ],
                                value='am'
                                )
col1 = dbc.Card(
    [
        dcc.Dropdown(
            id='scale-observation',
            options=[
                {'label': 'Induvidual Building', 'value': 'ind'},
                {'label': 'Wire All Buildings', 'value': 'wireb'},
                {'label': 'Wire', 'value': 'wire'},
                {'label': 'South/North Town', 'value': 'NotWire'},
                {'label': 'All of the Island', 'value': 'ri'}
            ],
            value='ri'
        ), mapColorDropDownMenu, dcc.Graph(id='map-graph', figure=fig_map)
    ],
    body=True
)
Menu3d = dcc.Dropdown(
    id='menu3D',
    options=[
        {'label': '3D Not Updated', 'value': 'No3D'},
        {'label': '3D Updated', 'value': 'Yes3D'},
    ],
    value='No3D'
)

contextualDropMenu = dcc.Dropdown(id='contextual-menu',
                                  options=[
                                      {'label': 'Age Income Bubbles',
                                          'value': 'aib'},
                                      {'label': 'Income Groups',
                                          'value': 'income'},
                                      {'label': 'Income Groups Census Categories',
                                          'value': 'incomeCensus'},
                                      {'label': 'Age Groups', 'value': 'age'},
                                      {'label': 'Apartment Cycles',
                                          'value': 'cycle'},
                                      {'label': 'Induvidual Apartments',
                                          'value': 'treemap'}
                                  ],
                                  value='aib'
                                  )


contextualCard = dbc.Card([contextualDropMenu, dcc.Graph(
    id='contextual-graph', figure=bubbleFig_)])

contextualCardProjDash = dbc.Card( dcc.Graph(
    id='contextual-graphProDash', figure=bubbleFig_))


sunBurstFigure = dbc.Card([dcc.Graph(id='sunBurst-graph', figure=figSunBurst)])

sunBurstFigureProjDash = dbc.Card(
    [dcc.Graph(id='sunBurst-graphProjDash', figure=figSunBurst)])

touchScreen = html.Div([dcc.Store(id='nothingObj'),
                        html.Table(
    [
        html.Tr([
                html.Td([html.Div(dbc.Card(html.H3(id="graph_ri"),
                                           style={'text-align': 'center'}, body=True))]),
                html.Td([html.Div(dbc.Card(html.H3(["Household in scope: Mean Age: Mean Income:"],
                                                   id="executive_sum_text"), style={'text-align': 'center'}, body=True))], colSpan='3')
                # html.Td([dbc.Card(dbc.CardBody([dcc.Input(id='figure_text', value='Figures', type='text'), html.Button('Download Figures', id='downloadB')])
                #                   )], style={'text-align': 'right'})

                ]),
        html.Tr(
            [
                html.Td(
                    [
                        dbc.Card([Menu3d, (html.Iframe(id='ifame-cell', height=f"{sim_plot.mapHeight1+40}px", width=f"{sim_plot.mapWidth}px",
                                                       src="https://technion-gis.maps.arcgis.com/apps/instant/3dviewer/index.html?appid=70a5849b08a643e188c1e082cfb579c4"))], body=True)
                    ], rowSpan='3'),
                html.Td([col1], rowSpan='3', style={
                    'border-style': 'solid', 'border-width': '0px', 'width': '400px'}),
                html.Td([dbc.Card(dbc.CardBody([
                        html.Div(dbc.Card(
                            dcc.Slider(id='year-slider', min=1976, max=2080, step=1, marks=yearListSlider, value=2000), style={"height": "100%"}, body=True),
                            #
                            style={'width': '74.5%','height':'100px', 'display': 'inline-block'}),
                        html.Div(dbc.Card(timeDropDownMenu, style={"height": "100%"}, body=True), style={
                                 'width': '24.5%', 'display': 'inline-block', 'vertical-align': 'top'})

                        ]))], colSpan='2', style={'border-style': 'solid', 'border-width': '0px'})
            ]
        ),
        html.Tr(
            [
                html.Td(dbc.Card(dbc.Card([cellTimeFigure]), body=True), colSpan='2', style={
                    'border-style': 'solid', 'border-width': '0px'})

            ]
        ),
        html.Tr(
            [
                html.Td(dbc.Card(sunBurstFigure, body=True), style={
                    'border-style': 'solid', 'border-width': '0px'}),
                html.Td(dbc.Card(contextualCard, body=True), style={
                    'border-style': 'solid', 'border-width': '0px'})

            ]
        )
    ],
    style={'border-collapse': 'collapse',
           'border-spacing': '0', 'width': '100%','font-size':'24px'}
)

])

projDash = html.Div([
    html.Table(
        [
            html.Tr([
                html.Td([html.Div(dbc.Card(html.H5(['All of the Island:'], id="bldYearProj"),
                                           style={'text-align': 'center'}, body=True))]),
                html.Td([html.Div(dbc.Card(html.H6(["In Score - Buidlings:0, HH:0, Tenant Cycles:0, Mean  Age:0, Mean Income:0"],
                                                   id="executiveSumTextProj"), style={'text-align': 'center'}, body=True))])

            ]),
            html.Tr(
                [
                    html.Td([dbc.Card(dbc.CardBody([
                        html.Div(dbc.Card(
                            dcc.Slider(id='yearSliderProj', min=1976, max=2080, step=1, marks=yearListSlider, value=2000), style={"height": "100%"}, body=True),
                            style={'width': '100%'})
                    ]))], colSpan='2', style={'border-style': 'solid', 'border-width': '0px'})
                ]
            ),
            html.Tr(
                [
                    html.Td(dbc.Card(dbc.Card([cellTimeFigureProjDash]), body=True), colSpan='2', style={
                        'border-style': 'solid', 'border-width': '0px'})
                ]
            ),
            html.Tr(
                [
                    html.Td(dbc.Card(sunBurstFigureProjDash, body=True), style={
                            'border-style': 'solid', 'border-width': '0px'}),
                    html.Td(dbc.Card(contextualCardProjDash, body=True), style={
                            'border-style': 'solid', 'border-width': '0px'})

                ]
            )

        ],
        style={'border-collapse': 'collapse',
               'border-spacing': '0', 'width': '100%'}

    ),
    dcc.Interval(
        id='interval-component_DashProj',
        interval=1*1500,  # in milliseconds
        n_intervals=0
    )

])

proj3D = html.Div([html.Table(
    [
        html.Tr(
            [
                html.Td(
                    [
                        dbc.Card([(html.Iframe(id='ifame-cellProj3D', height="1080px", width="580px",
                                               src="https://technion-gis.maps.arcgis.com/apps/instant/3dviewer/index.html?appid=70a5849b08a643e188c1e082cfb579c4"))], body=True)
                    ]),
                html.Td([dbc.Card(dcc.Graph(id='map-graphProj3D', figure=fig_map), body=True)], style={
                    'border-style': 'solid', 'border-width': '0px', "height":"1080px", 'width': '580px'}),
            ]
        ), dcc.Interval(
            id='interval-component_Dash3D',
            interval=1*1500,  # in milliseconds
            n_intervals=0
        )
    ],
    style={'border-collapse': 'collapse',
           'border-spacing': '0', 'width': '100%'}
)
])

Only3D = html.Div([html.Iframe(id='ifame-cellOnly3D', height="1200px", width="3500px",
                                              src="https://technion-gis.maps.arcgis.com/apps/instant/3dviewer/index.html?appid=70a5849b08a643e188c1e082cfb579c4"),dcc.Interval(
            id='interval_Only3D',
            interval=1*1500,  # in milliseconds
            n_intervals=0)
    ],style={'border-collapse': 'collapse',
          'border-spacing': '0', 'width': '100%'})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')


])

@app.callback([Output('bldYearProj', 'children'), Output('time-graphProjDash', 'figure'), Output('sunBurst-graphProjDash', 'figure'), Output('contextual-graphProDash', 'figure'), Output('executiveSumTextProj', 'children'), Output('yearSliderProj', 'value')],
              Input('interval-component_DashProj', 'n_intervals'))
def updateProjDash(n):
    #dbCSV = pd.read_csv('db.csv')
    with open('/home/ridigitaltwin/demo_2/assets/db.json') as json_file:
        dbCSV = json.load(json_file)
    BldName = dbCSV['BldName'][0]
    year_ = dbCSV['yearValue'][0]
    res_ = dbCSV['resolutionValue'][0]
    context_ = dbCSV['contextValue'][0]
    timeFigCategory_ = dbCSV['timeFigCategory'][0]
    # bldgYear = dbCSV['bldYearProj_'].values[0]
    # executiveText = dbCSV['executiveText'].values[0]
    #print('bldName', BldName)
    [bldgYear, res, timeFig, sunBurst, figContextual, executiveText] = TimeSunBurstContextFigure(
        BldName, year_, res_, context_, timeFigCategory_)
    timeFig.update_layout(width=1200, height=500,font=dict(size=24))
    sunBurst.update_layout(width=600, height=500)
    figContextual.update_layout(width=600, height=500,font=dict(size=24))
    return [bldgYear, timeFig, sunBurst, figContextual, executiveText, year_]

#interval-component_Only3D


@app.callback([ Output('ifame-cellOnly3D', 'src')], Input('interval_Only3D', 'n_intervals'))
def updateOnly3D_1(n):
    #toDB = pd.read_csv('db_maps.csv')
    with open('/home/ridigitaltwin/demo_2/assets/db_maps.json') as json_file:
        data = json.load(json_file)
    resolution_ = data['Resolution_'][0]
    menu_3d = data['menu_3d'][0]
    if (menu_3d == 'Yes3D'):
        url3D = getIframeURLfor3D(zoomto=resolution_)
    else:
        url3D = "https://technion-gis.maps.arcgis.com/apps/instant/3dviewer/index.html?appid=70a5849b08a643e188c1e082cfb579c4"
    return ([url3D])

@app.callback([Output('map-graphProj3D', 'figure'), Output('ifame-cellProj3D', 'src')], Input('interval-component_Dash3D', 'n_intervals'))
def updateProj3D(n):
    #toDB = pd.read_csv('db_maps.csv')
    with open('/home/ridigitaltwin/demo_2/assets/db_maps.json') as json_file:
        data = json.load(json_file)
    yearValue = data['yearValue'][0]
    mapCat = data['mapCat'][0]
    resolution_ = data['Resolution_'][0]
    menu_3d = data['menu_3d'][0]
    r = resultsAll1.getAffordableMarketPerYear3(yearValue)
    mapFigure = updateMapYear1(
        int(yearValue), rib.copy(), r, mapCat, zoomto=resolution_)
    mapFigure.update_layout(autosize=True, height=1080, width=580)
    if (menu_3d == 'Yes3D'):
        url3D = getIframeURLfor3D(zoomto=resolution_)
    else:
        url3D = "https://technion-gis.maps.arcgis.com/apps/instant/3dviewer/index.html?appid=70a5849b08a643e188c1e082cfb579c4"
    return ([mapFigure, url3D])


@ app.callback(
    [Output('map-graph', 'figure'), Output('ifame-cell', 'src')],
    [Input('year-slider', 'value'), Input('mapcolor-menu', 'value'), Input('graph_ri', 'children'), Input('menu3D', 'value')])
def update_map2(yearValue, cat, titleText, menu_3d):
    r = resultsAll1.getAffordableMarketPerYear3(yearValue)
    titleText2 = titleText.split(':')[0]
    mapFigure = updateMapYear1(
        int(yearValue), rib.copy(), r, cat, zoomto=titleText2)

    if (menu_3d == 'Yes3D'):
        url3D = getIframeURLfor3D(zoomto=titleText2)
    else:
        url3D = "https://technion-gis.maps.arcgis.com/apps/instant/3dviewer/index.html?appid=70a5849b08a643e188c1e082cfb579c4"

    #toDB = pd.DataFrame({'yearValue': [yearValue], 'mapCat': [
    #                    cat], 'Resolution_': [titleText2], 'menu_3d': [menu_3d]})
    #toDB.to_csv('db_maps.csv')
    data = {'yearValue': [yearValue], 'mapCat': [
        cat], 'Resolution_': [titleText2], 'menu_3d': [menu_3d]}
    with open('/home/ridigitaltwin/demo_2/assets/db_maps.json', 'w') as outfile:
        json.dump(data, outfile)
    return ([mapFigure, url3D])


# @ app.callback(
#     [Output('nothingObj', 'data')],
#     [Input('downloadB', 'n_clicks')],
#     [State('map-graph', 'figure'), State('time-graph', 'figure'), State('sunBurst-graph', 'figure'), State('contextual-graph', 'figure'), State('figure_text', 'value')])
# def update_output(n_clicks, mapData, graphTime, sunburstGraph, contextGraph, fileText):
#     if n_clicks != None:
#         now = datetime.datetime.now()
#         current_time = now.strftime("%Y%m%d_%H%M")
#         fig = go.Figure(data=mapData)
#         fig.write_html(f'exported/mapFig_{fileText}_{current_time}.html')
#         fig = go.Figure(data=graphTime)
#         fig.write_html(f'exported/timeFig_{fileText}_{current_time}.html')
#         fig = go.Figure(data=sunburstGraph)
#         fig.write_html(f'exported/sunburstFig_{fileText}_{current_time}.html')
#         fig = go.Figure(data=contextGraph)
#         fig.write_html(f'exported/contextFig_{fileText}_{current_time}.html')
#     return ['']


@ app.callback(
    [Output('graph_ri', 'children'), Output('scale-observation', 'value'), Output('time-graph', 'figure'),
     Output('sunBurst-graph', 'figure'), Output('contextual-graph', 'figure'), Output('executive_sum_text', 'children')],
    [Input('map-graph', 'clickData'), Input('year-slider', 'value'), Input('scale-observation', 'value'), Input('contextual-menu', 'value'), Input('time-menu', 'value')])
def update_graph_ri(clickData, yearValue, resolutionValue, contextValue, timeFigCategory):
    # executiveDefaultText = "In Score: Buildings: Households in Scope: Mean Age: Mean Income:"
    # executiveText = executiveDefaultText
    # #
    if (type(clickData) is not dict):
        BldName = 'None'
    else:
        BldName = clickData['points'][0]['customdata'][0]
    return TimeSunBurstContextFigure(BldName, yearValue, resolutionValue, contextValue, timeFigCategory)

    # return clickData['points']
# Update the index


def TimeSunBurstContextFigure(BldName, yearValue, resolutionValue, contextValue, timeFigCategory):
    currentData_ = resultsAll1.getAffordableMarketPerYear3(yearValue)
    currentData_.loc[currentData_['Building Name'] ==
                     'island house', 'Building Name'] = 'Island House'
    if (BldName == 'None'):
        # print_in_display('Nothing to Print')
        if (resolutionValue == 'ri'):
            if (timeFigCategory == 'am'):
                generalCopy = AllAffordable.copy()
                generalCopy = generalCopy.query(f"year<={yearValue}")
                fig = affordabilityTimeSeriesAgregattedGraph(
                    aw=generalCopy, mC=marketColor, aC=affordColor, title_='All of the Island')
            elif (timeFigCategory == 'leave'):
                re = allByYear.query(f"year<={yearValue}").copy()
                fig = sim_plot.reasulToLeaveByTime(
                    re, 'Reason for leaving for all of the Island')
            elif (timeFigCategory == 'life'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                fig = sim_plot.averageAgeByTime(
                    r, "Average Age and Life Expectancy for all of the Island")
            elif (timeFigCategory == 'ie'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                fig = sim_plot.incomeBurdenTime(
                    r, "Income and burden for all of the Island")
            elif (timeFigCategory == 'ageg'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                fig = sim_plot.ageGroupTimeGraph(
                    r, "Age Groups for all of the Island")
            else:
                r = allByYearStay.query(f"year<={yearValue}").copy()
                fig = sim_plot.incomeGroupTimeGraph(
                    r, "Income And Burden for all of the Island")
            sunBurstFig = sim_plot.sunburstGroupsAffordMarketYearColor3(
                currentData_.copy(), yearValue)
            if (contextValue == 'aib'):
                figContextual = sim_plot.bubbleAgeIncomeClass(
                    currentData_.copy(), yearValue, 'RI')
            elif(contextValue == 'income'):
                figContextual = sim_plot.incomeByGroupFigure(
                    currentData_.copy(), yearValue, 'RI')
            elif(contextValue == 'incomeCensus'):
                figContextual = sim_plot.incomeByGroupFigureCensus(
                    currentData_.copy(), yearValue, 'RI')
            elif(contextValue == 'age'):
                figContextual = sim_plot.ageByGroupFigure(
                    currentData_.copy(), yearValue, 'RI')
            elif(contextValue == 'cycle'):
                figContextual = sim_plot.AgentCycle(
                    currentData_.copy(), yearValue, 'RI')
            else:
                figContextual = sim_plot.treeMapIsland(
                    currentData_, f'Affordability and Income Agent Scale {yearValue} RI')
            executiveText = getCurrentScope(currentData_)

            # toDB = pd.DataFrame({'BldName': [BldName], 'bldYearProj_': [f'All of The Island: {yearValue}'], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
            #    resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]})
            # toDB.to_csv('db.csv')
            data = {'BldName': [BldName], 'bldYearProj_': [f'All of The Island: {yearValue}'], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
                resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]}
            with open('/home/ridigitaltwin/demo_2/assets/db.json', 'w') as outfile:
                json.dump(data, outfile)
            return ([f'All of The Island: {yearValue}', resolutionValue, fig, sunBurstFig, figContextual, executiveText])
        else:  # (resolutionValue in ['wire','wireb','NotWire']):
            if resolutionValue == 'NotWire':
                bldScope = 'Northtown & Southtown'
            else:
                bldScope = 'Wire'
            if (timeFigCategory == 'am'):
                if resolutionValue == 'wire':
                    rg = WireAffordable.copy()
                    rg = rg.query(f"year<={yearValue}")
                    title_ = 'Wire'
                    fig = affordabilityTimeSeriesAgregattedGraph(
                        aw=rg, mC=marketColor, aC=affordColor, title_=title_)
                elif resolutionValue == 'NotWire':
                    rg = NotWireAffordalbe.copy()
                    rg = rg.query(f"year<={yearValue}")
                    title_ = 'North Town and South Town'
                    fig = affordabilityTimeSeriesAgregattedGraph(
                        aw=rg, mC=marketColor, aC=affordColor, title_=title_)
                else:
                    fig = createAffordableInduvidualBldgs(
                        'Wire', 1500, 0.9, (f'Group=="WIRE" and year<{yearValue}'))
            elif (timeFigCategory in ['leave', 'life', 'ie', 'ageg', 'ig']):
                if resolutionValue == 'NotWire':
                    r = allByYearStay.query(f"year<={yearValue}").copy()
                    r = r.query('Group!="WIRE"').copy()
                    re = allByYear.query(f"year<={yearValue}").copy()
                    re = re.query('Group!="WIRE"').copy()
                    groupBuildings = 'Northtown Southtown'
                else:
                    r = allByYearStay.query(f"year<={yearValue}").copy()
                    r = r.query('Group=="WIRE"').copy()
                    re = allByYear.query(f"year<={yearValue}").copy()
                    re = re.query('Group=="WIRE"').copy()
                    groupBuildings = 'Wire'

                if (timeFigCategory == 'leave'):

                    fig = sim_plot.reasulToLeaveByTime(
                        re, f'Reason for leaving for {groupBuildings} Buidlings')
                elif (timeFigCategory == 'life'):

                    fig = sim_plot.averageAgeByTime(
                        r, f"Average Age and Life Expectancy for {groupBuildings} Buidlings")
                elif (timeFigCategory == 'ie'):
                    fig = sim_plot.incomeBurdenTime(
                        r, f"Income and burden for {groupBuildings} Buidlings")
                elif (timeFigCategory == 'ageg'):
                    fig = sim_plot.ageGroupTimeGraph(
                        r, f"Age Groups for {groupBuildings} Buidlings")
                else:

                    fig = sim_plot.incomeGroupTimeGraph(
                        r, f"Income And Burden for {groupBuildings} Buidlings")
            if resolutionValue == 'NotWire':
                groupBuildings = 'Northtown Southtown'
                currentDataWire = currentData_.query('Group!="WIRE"').copy()
            else:
                groupBuildings = 'Wire'
                currentDataWire = currentData_.query('Group=="WIRE"').copy()
            sunBurstFig = sim_plot.sunburstGroupsAffordMarketYearColor3(
                currentDataWire, yearValue)
            if (contextValue == 'aib'):
                figContextual = sim_plot.bubbleAgeIncomeClass(
                    currentDataWire.copy(), yearValue, groupBuildings)
            elif(contextValue == 'income'):
                figContextual = sim_plot.incomeByGroupFigure(
                    currentDataWire.copy(), yearValue, groupBuildings)
            elif(contextValue == 'incomeCensus'):
                figContextual = sim_plot.incomeByGroupFigureCensus(
                    currentDataWire.copy(), yearValue, groupBuildings)
            elif(contextValue == 'age'):
                figContextual = sim_plot.ageByGroupFigure(
                    currentDataWire.copy(), yearValue, groupBuildings)
            elif(contextValue == 'cycle'):
                figContextual = sim_plot.AgentCycle(
                    currentDataWire.copy(), yearValue, groupBuildings)
            else:
                figContextual = sim_plot.treeMapIsland(currentDataWire.copy(
                ), f'Affordability and Income Agent Scale {yearValue} {groupBuildings}')
            executiveText = getCurrentScope(currentDataWire)

            # toDB = pd.DataFrame({'BldName': [BldName], 'bldYearProj_': [f'{bldScope}: {yearValue}'], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
            #    resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]})
            # toDB.to_csv('db.csv')
            data = {'BldName': [BldName], 'bldYearProj_': [f'{bldScope}: {yearValue}'], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
                resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]}
            with open('/home/ridigitaltwin/demo_2/assets/db.json', 'w') as outfile:
                json.dump(data, outfile)
            return ([f'{bldScope}: {yearValue}', resolutionValue, fig, sunBurstFig, figContextual, executiveText])
    else:
        if (resolutionValue == 'ri'):

            if (timeFigCategory == 'am'):
                generalCopy = AllAffordable.copy()
                generalCopy = generalCopy.query(f"year<={yearValue}")
                fig = affordabilityTimeSeriesAgregattedGraph(
                    aw=generalCopy, mC=marketColor, aC=affordColor, title_='All of the Island')
            elif (timeFigCategory == 'leave'):
                re = allByYear.query(f"year<={yearValue}").copy()
                fig = sim_plot.reasulToLeaveByTime(
                    re, 'Reason for leaving for all of the Island')
            elif (timeFigCategory == 'life'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                fig = sim_plot.averageAgeByTime(
                    r, "Average Age and Life Expectancy for all of the Island")
            elif (timeFigCategory == 'ie'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                fig = sim_plot.incomeBurdenTime(
                    r, "Income and burden for all of the Island")
            elif (timeFigCategory == 'ageg'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                fig = sim_plot.ageGroupTimeGraph(
                    r, "Age Groups for all of the Island")
            else:
                r = allByYearStay.query(f"year<={yearValue}").copy()
                fig = sim_plot.incomeGroupTimeGraph(
                    r, "Income And Burden for all of the Island")

            sunBurstFig = sim_plot.sunburstGroupsAffordMarketYearColor3(
                currentData_.copy(), yearValue)
            if (contextValue == 'aib'):
                figContextual = sim_plot.bubbleAgeIncomeClass(
                    currentData_, yearValue, 'RI')
            elif(contextValue == 'income'):
                figContextual = sim_plot.incomeByGroupFigure(
                    currentData_, yearValue, 'RI')
            elif(contextValue == 'incomeCensus'):
                figContextual = sim_plot.incomeByGroupFigureCensus(
                    currentData_, yearValue, 'RI')
            elif(contextValue == 'age'):
                figContextual = sim_plot.ageByGroupFigure(
                    currentData_, yearValue, 'RI')
            elif(contextValue == 'cycle'):
                figContextual = sim_plot.AgentCycle(
                    currentData_.copy(), yearValue, 'RI')
            else:
                figContextual = sim_plot.treeMapIsland(
                    currentData_.copy(), f'Affordability and Income Agent Scale {yearValue} RI')

            executiveText = getCurrentScope(currentData_)

            # toDB = pd.DataFrame({'BldName': [BldName], 'bldYearProj_': [f'All of The Island: {yearValue}'], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
            #    resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]})
            # toDB.to_csv('db.csv')
            data = {'BldName': ['None'], 'bldYearProj_': [f'All of The Island: {yearValue}'], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
                resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]}
            with open('/home/ridigitaltwin/demo_2/assets/db.json', 'w') as outfile:
                json.dump(data, outfile)
            return ([f'All of The Island: {yearValue}', resolutionValue, fig, sunBurstFig, figContextual, executiveText])
        elif (resolutionValue in ['wire', 'NotWire', 'wireb']):
            if resolutionValue == 'NotWire':
                bldScope = 'Northtown & Southtown'
            else:
                bldScope = 'Wire'
            if (timeFigCategory == 'am'):
                if resolutionValue == 'wire':
                    rg = WireAffordable.copy()
                    rg = rg.query(f"year<={yearValue}")
                    title_ = 'Wire'
                    fig = affordabilityTimeSeriesAgregattedGraph(
                        aw=rg, mC=marketColor, aC=affordColor, title_=title_)
                elif resolutionValue == 'NotWire':
                    rg = NotWireAffordalbe.copy()
                    rg = rg.query(f"year<={yearValue}")
                    title_ = 'North Town and South Town'
                    fig = affordabilityTimeSeriesAgregattedGraph(
                        aw=rg, mC=marketColor, aC=affordColor, title_=title_)
                else:
                    fig = createAffordableInduvidualBldgs(
                        'Wire', 1500, 0.9, (f'Group=="WIRE" and year<{yearValue}'))

            elif (timeFigCategory in ['leave', 'life', 'ie', 'ageg', 'ig']):
                if resolutionValue == 'NotWire':
                    r = allByYearStay.query(f"year<={yearValue}").copy()
                    r = r.query('Group!="WIRE"').copy()
                    re = allByYear.query('Group!="WIRE"').copy()
                    re = re.query(f"year<={yearValue}").copy()
                    groupBuildings = 'Northtown Southtown'
                else:
                    r = allByYearStay.query(f"year<={yearValue}").copy()
                    r = r.query('Group=="WIRE"').copy()
                    re = allByYear.query('Group=="WIRE"').copy()
                    re = re.query(f"year<={yearValue}").copy()
                    groupBuildings = 'Wire'
                if (timeFigCategory == 'leave'):
                    fig = sim_plot.reasulToLeaveByTime(
                        re, f'Reason for leaving for {groupBuildings} Buidlings')
                elif (timeFigCategory == 'life'):
                    fig = sim_plot.averageAgeByTime(
                        r, f"Average Age and Life Expectancy for {groupBuildings} Buidlings")
                elif (timeFigCategory == 'ie'):

                    fig = sim_plot.incomeBurdenTime(
                        r, f"Income and burden for {groupBuildings} Buidlings")
                elif (timeFigCategory == 'ageg'):

                    fig = sim_plot.ageGroupTimeGraph(
                        r, f"Age Groups for {groupBuildings} Buidlings")
                else:
                    fig = sim_plot.incomeGroupTimeGraph(
                        r, f"Income And Burden for {groupBuildings} Buidlings")
            if resolutionValue == 'NotWire':
                groupBuildings = 'Northtown Southtown'
                currentDataWire = currentData_.query('Group!="WIRE"').copy()
            else:
                groupBuildings = 'Wire'
                currentDataWire = currentData_.query('Group=="WIRE"').copy()
            sunBurstFig = sim_plot.sunburstGroupsAffordMarketYearColor3(
                currentDataWire, yearValue)
            if (contextValue == 'aib'):
                figContextual = sim_plot.bubbleAgeIncomeClass(
                    currentDataWire, yearValue, groupBuildings)
            elif(contextValue == 'income'):
                figContextual = sim_plot.incomeByGroupFigure(
                    currentDataWire, yearValue, groupBuildings)
            elif(contextValue == 'incomeCensus'):
                figContextual = sim_plot.incomeByGroupFigureCensus(
                    currentDataWire, yearValue, groupBuildings)
            elif(contextValue == 'age'):
                figContextual = sim_plot.ageByGroupFigure(
                    currentDataWire, yearValue, groupBuildings)
            elif(contextValue == 'cycle'):
                figContextual = sim_plot.AgentCycle(
                    currentDataWire, yearValue, groupBuildings)
            else:
                figContextual = sim_plot.treeMapIsland(
                    currentDataWire, f'Affordability and Income Agent Scale {yearValue} {groupBuildings}')
            executiveText = getCurrentScope(currentDataWire)

            # toDB = pd.DataFrame({'BldName': [BldName], 'bldYearProj_': [f'{bldScope}: {yearValue}'], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
            #     resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]})
            # toDB.to_csv('db.csv')
            data = {'BldName': ['None'], 'bldYearProj_': [f'{bldScope}: {yearValue}'], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
                resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]}
            with open('/home/ridigitaltwin/demo_2/assets/db.json', 'w') as outfile:
                json.dump(data, outfile)
            return ([f'{bldScope}: {yearValue}', resolutionValue, fig, sunBurstFig, figContextual, executiveText])
        else:
            if (timeFigCategory == 'am'):
                r = allAffordableByBldg[allAffordableByBldg['Building Name'] == f'{BldName}'].copy(
                )
                r = r.query(f"year<={yearValue}").copy()
                r.reset_index(inplace=True, drop=True)
                fig = affordabilityTimeSeriesAgregattedGraph(
                    aw=r, mC=marketColor, aC=affordColor, title_=BldName)
            elif (timeFigCategory == 'leave'):
                re = allByYear.query(f"year<={yearValue}").copy()
                re = re.query(f'`Building Name`=="{BldName}"').copy()
                fig = sim_plot.reasulToLeaveByTime(
                    re, f'Reason for leaving for {BldName}')
            elif (timeFigCategory == 'life'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                r = r.query(f'`Building Name`=="{BldName}"').copy()
                fig = sim_plot.averageAgeByTime(
                    r, f"Average Age and Life Expectancy for {BldName}")
            elif (timeFigCategory == 'ie'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                r = r.query(f'`Building Name`=="{BldName}"').copy()
                fig = sim_plot.incomeBurdenTime(
                    r, f"Income and burden for {BldName}")
            elif (timeFigCategory == 'ageg'):
                r = allByYearStay.query(f"year<={yearValue}").copy()
                r = r.query(f'`Building Name`=="{BldName}"').copy()
                fig = sim_plot.ageGroupTimeGraph(
                    r, f"Age Groups for {BldName}")
            else:
                r = allByYearStay.query(f"year<={yearValue}").copy()
                r = r.query(f'`Building Name`=="{BldName}"').copy()
                fig = sim_plot.incomeGroupTimeGraph(
                    r, f"Income And Burden for {BldName}")
            currentDataBldg = currentData_.query(
                f'`Building Name`=="{BldName}"').copy()
            sunBurstFig = sim_plot.sunburstGroupsAffordMarketYearColor3(
                currentDataBldg, yearValue)
            if (contextValue == 'aib'):
                figContextual = sim_plot.bubbleAgeIncomeClass(
                    currentDataBldg, yearValue, 'Building')
            elif(contextValue == 'income'):
                figContextual = sim_plot.incomeByGroupFigure(
                    currentDataBldg, yearValue, 'Building')
            elif(contextValue == 'incomeCensus'):
                figContextual = sim_plot.incomeByGroupFigureCensus(
                    currentDataBldg, yearValue, 'Building')
            elif(contextValue == 'age'):
                figContextual = sim_plot.ageByGroupFigure(
                    currentDataBldg, yearValue, 'Building')
            elif(contextValue == 'cycle'):
                figContextual = sim_plot.AgentCycle(
                    currentDataBldg, yearValue, 'Building')
            else:
                figContextual = sim_plot.treeMapBuilding(
                    currentDataBldg, f'Affordability and Income Agent Scale {yearValue} Building')
            executiveText = getCurrentScope(currentDataBldg)
            # toDB = pd.DataFrame({'BldName': [BldName], 'bldYearProj_': [BldName+": " + str(yearValue)], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
            #     resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]})
            # toDB.to_csv('db.csv')
            data = {'BldName': [BldName], 'bldYearProj_': [BldName+": " + str(yearValue)], 'executiveText': [executiveText], 'yearValue': [yearValue], 'resolutionValue': [
                resolutionValue], 'contextValue': [contextValue], 'timeFigCategory': [timeFigCategory]}
            with open('/home/ridigitaltwin/demo_2/assets/db.json', 'w') as outfile:
                json.dump(data, outfile)
            return ([BldName+": " + str(yearValue), resolutionValue, fig, sunBurstFig, figContextual, executiveText])


@ app.callback(dash.dependencies.Output('page-content', 'children'),
               [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/touchScreen':
        return touchScreen
    elif pathname == '/ProjDash':
        return projDash
    elif pathname == '/Proj3D':
        return proj3D
    elif pathname == '/Only3D':
        return Only3D
        #return touchScreen
    else:
        return touchScreen
    # You could also return a 404 "URL not found" page here


# Run app and display result inline in the notebook
if __name__ == '__main__':
    # app.run_server(debug=True, host='127.0.0.1',
    # suppress_callback_exceptions=True)
    app.run_server(debug=False, dev_tools_ui=False,
                   dev_tools_props_check=False, host='127.0.0.1')
