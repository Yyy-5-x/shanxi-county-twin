import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import joblib
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from syn_sample import generate_syn_sample, validate_syn_sample
st.set_page_config(page_title="山西县域经济数字孪生原型", layout="wide")
st.title("山西县域经济数字孪生原型")

#数据集导入
st.markdown("数据集导入")
uploaded_file = st.file_uploader("上传CSV数据集", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"已读取上传数据集，共{df.shape[0]}行 {df.shape[1]}列")
else:
    df=pd.read_csv("山西县域面板数据.csv")
    st.info("未上传文件，加载本地山西县域面板数据集；上传其他CSV即可切换为其他地区孪生")

df["gdp相对误差"] = np.where(np.abs(df["gdp相对误差"]) < 1e-6, 0, df["gdp相对误差"])

#模型训练
st.markdown("模型管理")
upload_model_file = st.file_uploader("上传已保存的模型.pkl", type=["pkl"])

if upload_model_file is not None:
    twin_model = joblib.load(io.BytesIO(upload_model_file.read()))
    st.success("成功加载已有模型，无需重新训练")
else:
    st.subheader("模型选择")
    model_choice = st.selectbox("选择情景仿真模型", ["线性回归", "随机森林回归"])

    X = df[["城镇化率", "第一产业占gdp比重", "财政自给率"]].copy().dropna()
    y = df["GDP"].copy().loc[X.index]

    if model_choice == "线性回归":
        twin_model = LinearRegression()
    else:
        twin_model = RandomForestRegressor(n_estimators=100, random_state=42)
    twin_model.fit(X, y)
    y_pred_train = twin_model.predict(X)
    r2 = r2_score(y, y_pred_train)
    st.info(f"当前 {model_choice}，训练集R² = {r2:.4f}")
    st.info("模型已根据当前数据集完成训练，可进行情景仿真推演")
    # 下载模型按钮
    buffer = io.BytesIO()
    joblib.dump(twin_model, buffer)
    buffer.seek(0)
    st.download_button(
        label="下载当前训练好的模型(.pkl)",
        data=buffer,
        file_name="县域孪生推演模型.pkl",
        mime="application/octet-stream"
    )

#三个标签页
tab1, tab2, tab3 = st.tabs([
    "真实历史数据查询",
    "数字孪生情景仿真推演",
    "硅基样本生成与有效性校验"
])

# Tab1：真实历史数据查询
with tab1:
    with st.sidebar:
        st.header("筛选条件")
        year_options = sorted(df["year"].unique())
        choose_year = st.selectbox("选择年份", year_options)
        county_options = sorted(df["county_name"].unique())
        choose_county = st.multiselect("选择县域", county_options, default=county_options[:5])

    filter_data = df[(df["year"] == choose_year) & (df["county_name"].isin(choose_county))].copy()
    st.subheader("筛选结果数据表")
    st.dataframe(filter_data, use_container_width=True)

    csv_out = filter_data.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载筛选结果CSV", data=csv_out, file_name=f"县域真实数据_{choose_year}.csv")

    if len(filter_data) > 0:
        c1, c2 = st.columns(2)
        with c1:
            fig_gdp = px.bar(filter_data, x="county_name", y="GDP", title="各县GDP对比")
            st.plotly_chart(fig_gdp, use_container_width=True)
        with c2:
            fig_err = px.histogram(filter_data, x="gdp相对误差", title="GDP相对误差分布")
            st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("""
> 说明：本模块复刻县域真实历史状态，属于数字阴影；
> gdp_rel_error为多源融合相对误差，极小数值为浮点数舍入误差。
""")

# Tab2：孪生情景仿真推演
with tab2:
    st.subheader("情景仿真推演：修改参数，模拟县域经济变化")
    sel_county_sim = st.selectbox("选择仿真县域", sorted(df["county_name"].unique()))
    base_row = df[df["county_name"] == sel_county_sim].iloc[0]
    new_urban = st.number_input("修改城镇化率", min_value=0.0, max_value=1.0, value=float(base_row["城镇化率"]))
    new_pri = st.number_input("修改第一产业占比", min_value=0.0, max_value=1.0, value=float(base_row["第一产业占gdp比重"]))
    new_fin = st.number_input("修改财政自给率", min_value=0.0, max_value=1.0, value=float(base_row["财政自给率"]))
    if st.button("运行孪生仿真推演"):
        X_sim = pd.DataFrame([[new_urban, new_pri, new_fin]],
                             columns=["城镇化率", "第一产业占gdp比重", "财政自给率"])
        sim_gdp = twin_model.predict(X_sim)[0]
        real_gdp = base_row["GDP"]
        sim_rel_error = (sim_gdp-real_gdp) / real_gdp

        st.markdown(f"""
推演结果
仿真模型输出GDP：{sim_gdp:.2f} 万元
该县域真实年鉴GDP：{real_gdp:.2f} 万元
仿真相对误差：{sim_rel_error:.6f}
""")
        if abs(sim_rel_error) < 0.1:
            st.success("仿真结果保真度较好")
        else:
            st.warning("仿真偏差较大，参考意义有限")

        st.info("情景推演为虚拟模拟结果，不代表真实发生的现实；相对误差用于评估孪生模型虚实映射保真程度。")

# Tab3：硅基样本生成与有效性校验
with tab3:
    st.subheader("硅基虚拟样本生成与有效性")
    n_syn = st.number_input("生成硅基样本数量", min_value=100, max_value=2000, value=500)
    if st.button("一键生成硅基样本并校验"):
        df_rename = df.rename(columns={
            "城镇化率": "urban_rate",
            "第一产业占gdp比重": "pri_ratio",
            "财政自给率": "finance_self_support",
            "人均gdp": "gdp_per_capita"
        })
        real_sub, syn_sub = generate_syn_sample(df_rename, n_sample=n_syn)
        validate_res = validate_syn_sample(real_sub, syn_sub)
        st.markdown("校验结果表")
        validate_res = validate_syn_sample(real_sub, syn_sub)
        validate_df = pd.DataFrame(validate_res).T
        validate_df.index = [
            "城镇化率",
            "第一产业占gdp比重",
            "财政自给率",
            "人均gdp"
        ]
        rename_col_dict = {
            "ks_statistic": "KS统计量",
            "ks_pvalue": "KS检验p值",
            "real_mean": "真实样本均值",
            "syn_mean": "硅基样本均值",
            "mean_relative_error": "均值相对偏差"
        }
        validate_df = validate_df.rename(columns=rename_col_dict)
        validate_df = validate_df.round(4)
        st.dataframe(validate_df)
        st.markdown("样本分布对比图(城镇化率)")
        fig_dist = px.histogram(pd.concat([real_sub.
                                      assign(type="真实样本"), syn_sub.
                                      assign(type="硅基样本")]),
                            x="urban_rate", color="type",
                            barmode="overlay", opacity=0.6
                            )

        st.plotly_chart(fig_dist, use_container_width = True)
        csv_syn = syn_sub.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载硅基样本CSV", data=csv_syn, file_name="硅基虚拟样本.csv")
        st.markdown("""
        校验说明:
        1.KS检验p值 > 0.05: 真实样本与硅基样本分布无显著差异;
        2.均值、方差相对偏差越小, 样本统计保真度越高;
        3.硅基样本仅作为辅助补充, 不可完全替代真实统计样本
        """)
        st.markdown("""
        原型整体说明(陶飞五维模型映射):
        1.物理实体PE:县域真实经济系统
        2.孪生数据DD:多源融合面板数据集,含gdp相对误差真值校验
        3.虚拟模型VE:回归情景推演模型+ Copula硅基样本生成模型
        4.连接CN:数据上传、模型调用、界面交互
        5.服务系统Ss:查询、情景仿真、硅基样本生成校验、结果导出
        本原型支持网页上传外部CSV数据集,无需修改代码,即可快速构建其他地区县域经济数字孪生原型;支持模型下载保存、本地模型上传加载
        """)
