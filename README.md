# CMU Interactive Data Science Final Project

* **Online URL**: https://share.streamlit.io/cmu-ids-2020/fp-null-pointer-1/main
* **Team members**:
  * Contact person: Tianyang Zhan tzhan@andrew.cmu.edu
  * Ruoxin Huang ruoxinh@andrew.cmu.edu
  * Amanda Xu xinyix2@andrew.cmu.edu
  * Ziming He zimingh@andrew.cmu.edu
* **Track**: Model (one of Narrative, Model, or Interactive Visualization/Application)

## Build Instructions
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If you encounter the following error, try install the **ibgtk2.0-dev** package to your system as described [here](https://stackoverflow.com/questions/53359649/from-cv2-import-importerror-libgthread-2-0-so-0-cannot-open-shared-object-f?noredirect=1&lq=1).
```bash
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
```

## Work distribution
The work was split equally among the team members. Ruoxin and Amanda worked on the visualization part of the project. The responsibilities include: researching relevant research work and news articles related to COVID-19 and face masks, collecting and preprocessing datasets for visualizations, designing and implementing interactive plots to answer our research questions, and optimizing the UI and interaction of the application. Tianyang and Ziming worked on the model part of the project. The work includes researching face/mask detection models and frameworks, designing and implementing data pipelines for the system, integrating multiple pre-trained deep learning models, and developing a statistical model for safety level measurements.

In general, the development process of this project was smooth. All team members collaborated with each other effectively. We held weekly meetings on Zoom where the team updated, presented and reviewed the recent progress and discussed the ideas on different parts of the project. During the development of this project, we also researched and reviewed various journalistic and academic sources on the topic of COVID-19 and face mask policies. We are convinced that raising awareness of mask-wearing is critical for slowing down and preventing the spread of COVID. Therefore, we believe our work is meaningful, and can have real world implications. In addition, we believe our experience in this course has helped us improve our abilities to develop an interactive application that can reveal and communicate information in data in a clear, effective, and unbiased manner.

## Deliverables

### Project Proposal
* Proposal: [link](https://github.com/CMU-IDS-2020/fp-null-pointer-1/blob/main/Proposal.md)

### Design Review
* Design Review Notes: [note](https://github.com/CMU-IDS-2020/fp-null-pointer-1/blob/main/Design_Review.md)
* Demo Video: [video](https://drive.google.com/file/d/1mrhgV_-yGcusL5ZiUPzEE5wssBDSTUa_/view?usp=sharing)

### Final Report and Video
* Final Report: [report](https://github.com/CMU-IDS-2020/fp-null-pointer-1/blob/main/Final_Report.md)
* Final Demo Video: [video](https://drive.google.com/file/d/14BmtF6ckPxxoV3ReM3NApDO6dHwX3vo8/view?usp=sharing)

### Final deliverables

- [X] All code for the project should be in the repo.
- [X] A 5 minute video demonstration.
- [X] Update Readme according to Canvas instructions.
- [X] A detailed project report. The contact should submit the video and report as a PDF on Canvas.
